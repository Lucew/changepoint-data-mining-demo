# inspired by https://dash.plotly.com/interactive-graphing
import logging
import time

from dash import Dash, dcc, html, Input, Output, callback, State, register_page
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import rankdata as scipyrank

import util.load_data as utl
from GLOBALS import *
import util.cache_registry as ucache


# register the page to our application
register_page(__name__, path="/correlate")
logger = logging.getLogger("frontend-logger")


styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll',
            'color': 'black'
        },
        'div': {
            'padding': '.3rem',
            'width': '80%',
            'margin': 'auto',
            'boxShadow': 'dimgrey 4px 4px 2px',
            'border-radius': '10px',
            'backgroundColor': 'white',
            'marginTop': '1rem',
        },
        'dropdown': {
            'margin': 'auto',
            'width': '100%',
            'border-radius': '10px',
            'color': 'black'
        }
    }


@ucache.lru_cache(maxsize=1)
def get_score_information(session_id: str, folder_name: str):
    start = time.perf_counter()

    # get the scores from memory to preprocess them
    scores, _, _, _, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # group the scores by their window size
    scores = {name: df.set_index('timestamp').sort_index().groupby('window') for name, df in scores.items()}

    # get the minimum, maximum, mean and std of each score
    score_information = pd.DataFrame(index=pd.MultiIndex.from_tuples([(name, ws)
                                                                      for name in scores.keys()
                                                                      for ws in scores[name].groups.keys()]),
                                     columns=['Minimum', 'Maximum', 'Mean', 'Median', 'Std'])
    for name, grouped_df in tqdm(scores.items(), desc='Compute the Information'):
        for ws in grouped_df.groups.keys():
            df = grouped_df.get_group(ws)
            score_information.loc[(name, ws), :] = [df['value'].min(), df['value'].max(), df['value'].mean(),
                                                    df['value'].median(), df['value'].std()]

    # get the window sizes we computed
    scores_grouped = scores
    logger.info(f"[{__name__}] Preprocessed data in {time.perf_counter() - start:0.2f} s.")
    return scores_grouped, score_information


@callback(
    Output('selected-data', 'children'),
    Output('hidden-graph1', 'figure'),
    Output('hidden-graph1-container', 'hidden'),
    Output('hidden-graph2-container', 'hidden'),
    Input("session-id", "data"),
    Input("folder-name", "data"),
    Input('basic-interactions', 'selectedData'),
    Input('signal-select', 'value'),
    Input('color-select', 'value'),
    Input('window-select', 'value'),
    Input('method-select', 'value'),)
def display_selected_data(session_id: str, folder_name: str,
                          selected_data, selected_signal, selected_color_choice, window_size, selection_method):

    # check whether there is something selected
    if not selected_data or not selected_data['points']:
        return "", {}, True, True

    # get the minimum and maximum of all the selected points
    min_idx = min(pd.Timestamp(point['x']) for point in selected_data['points'])
    max_idx = max(pd.Timestamp(point['x']) for point in selected_data['points']) + pd.Timedelta(milliseconds=1)

    # get the changepoint signal during this change
    cp_similarities, names = find_others(session_id, folder_name, min_idx, max_idx, selected_signal, window_size, selection_method)

    # construct the result string
    res_str = [f"Points from signal {selected_signal} are selected from Index: {min_idx} to {max_idx}.\n"]
    # res_str += [names[idx] for idx in res]

    # get the correlation values for the raw signals
    _, _, _, _, _, _, signal_correlation = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # get the correlation values for the selected signal
    corr_vals = signal_correlation.loc[names, selected_signal].abs().to_numpy()

    # make a figure
    # get_rand(len(cp_similarities))
    fig = px.scatter(pd.DataFrame({'Signal Corr.': corr_vals,
                                   'cp_similarity': cp_similarities,
                                   'name': names,
                                   'Block+Turbine': [col[1:3] for col in names],
                                   'Block': [col[1:2] for col in names],
                                   'Component': [col[3:6] for col in names],
                                   'Turbine': [col[2:3] for col in names]}),
                     x='cp_similarity',
                     y='Signal Corr.',
                     color=selected_color_choice,
                     hover_data=['cp_similarity', 'name'],
                     marginal_x="histogram",
                     title=f"CP Similarity for Signal {selected_signal} from index {min_idx} to {max_idx} ({max_idx-min_idx}).")

    # make the histogram stacked
    # fig.update_layout(barmode='group')
    fig.update_layout(transition_duration=500)
    fig.update_layout(xaxis_title="Change Point Similarity", yaxis_title="Complete Signal Correlation")
    return "\n".join(res_str), fig, False, True


# make a python function that goes through all other signals and finds the most interesting ones
def find_others(session_id: str, folder_name: str, start, end, selected_signal, window_size, selection_method):
    _start = time.perf_counter()

    # get the data
    scores_grouped, score_information = get_score_information(session_id, folder_name)
    signal_names = list(scores_grouped.keys())

    # get the timezone of the selected signal
    tz = scores_grouped[selected_signal].get_group(window_size).index.tz

    # set the timezone of the scores
    start = start.tz_localize(tz)
    end = end.tz_localize(tz)

    # get the selected signal data
    signal = scores_grouped[selected_signal].get_group(window_size).loc[start:end, 'value'].to_numpy()

    # get the other signals with the correct window size and put them into a numpy array
    other_array = np.empty((len(signal_names) - 1, signal.shape[0]))
    signal_names_new = []
    idx = 0
    for name in signal_names:
        if name == selected_signal or name not in scores_grouped:
            continue

        # check how many values we have
        other_sig = scores_grouped[name].get_group(window_size).loc[start:end, 'value'].to_numpy()
        if other_sig.shape != signal.shape:
            continue

        # fill in the data
        signal_names_new.append(name)
        other_array[idx, :] = other_sig
        idx += 1
    signal_names = signal_names_new
    other_array = other_array[:idx, :]

    # make a decision based on the selection method
    if selection_method == 'Pearson Correlation':  # find the signal with the highest correlation in this time
        # process the original signal
        signal_mean = np.mean(signal)
        signal -= signal_mean
        signal_std = np.sqrt(np.sum(np.square(signal)))

        # compute the mean of all signals
        signal_means = np.mean(other_array, axis=1)

        # subtract the mean from the array
        other_array -= signal_means[:, None]

        # compute the standard deviations of the mean
        other_std = np.sqrt(np.sum(np.square(other_array), axis=1)) + np.finfo('float').eps

        # compute the covariance
        corr = (other_array @ signal[:, None])/(other_std * signal_std)[:, None]
        corr = corr[:, 0]

    elif selection_method == 'Activity':  # find the signal with the highest normalized activity

        # get the data for the min-max-normalization
        other_array_min = np.array([score_information.loc[(name, window_size), 'Minimum'] for name in signal_names])
        other_array_max = np.array([score_information.loc[(name, window_size), 'Maximum'] for name in signal_names])

        # make the min-max-normalization
        other_divisor = other_array_max-other_array_min
        other_divisor[other_divisor <= np.finfo('float').eps*10] = 1
        other_array = (other_array - other_array_min[:, None])/other_divisor[:, None]

        # check for the highest activity value
        corr = np.max(other_array, axis=1)

    elif selection_method == 'Spearman Correlation':  # find the signal with the highest spearman correlation

        # compute the spearman r
        # checked = np.array([scipyr(a=signal, b=other_array[idx, :]).statistic for idx in range(other_array.shape[0])])

        # convert the signals into ranks (https://stackoverflow.com/a/5284703, https://stackoverflow.com/a/29059978)
        signal = scipyrank(signal).astype('float')
        other_array = scipyrank(other_array, axis=1).astype('float')

        # process the original signal
        signal_mean = np.mean(signal)
        signal -= signal_mean
        signal_std = np.sqrt(np.sum(np.square(signal)))

        # compute the mean of all signals
        signal_means = np.mean(other_array, axis=1)

        # subtract the mean from the array
        other_array -= signal_means[:, None]

        # compute the standard deviations of the mean
        other_std = np.sqrt(np.sum(np.square(other_array), axis=1)) + np.finfo('float').eps

        # compute the covariance
        corr = (other_array @ signal[:, None]) / (other_std * signal_std)[:, None]
        corr = corr[:, 0]

    else:
        raise NotImplementedError(f'Method {selection_method} not yet implemented.')
    logger.info(f"[{__name__}] Finding other signals in {time.perf_counter() - _start:0.3f} s. Parameters: {selection_method=}, {start=}, {end=}, {selected_signal=}, {window_size=}.")
    return corr, signal_names


@ucache.lru_cache(10)
def get_rand(length: int):
    return np.random.uniform(-1, 1, length)


def make_signal_figure(signal_df, score_df, signal_name):

    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # make the signal plot
    fig.add_trace(
        go.Scatter(x=signal_df.index, y=signal_df['value'], name=signal_name, mode='lines+markers'),
        secondary_y=False,
    )

    # make the score plot
    fig.add_trace(
        go.Scatter(x=score_df.index, y=score_df['value'], name="Change Score"),
        secondary_y=True,
    )

    # update the layout
    fig.update_traces(marker_size=5)
    fig.update_layout(transition_duration=500)
    fig.update_layout(dragmode='select')
    return fig



@callback(
    Output('basic-interactions', 'figure', allow_duplicate=True),
    Output('window-select', 'options'),
    Output('window-select', 'value'),
    Input("session-id", "data"),
    Input("folder-name", "data"),
    Input('signal-select', 'value'),
    prevent_initial_call='initial_duplicate'
    )
def change_signal(session_id: str, folder_name: str, selected_signal):

    # get the signal data and grouped score data
    _, _, _, _, _, raw_signals_grouped, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))
    scores_grouped, score_information = get_score_information(session_id, folder_name)

    # get the signal and score data
    signal = raw_signals_grouped.get_group(selected_signal)
    scores = scores_grouped[selected_signal]

    # get the number of windows we computed for the signal
    ws_sizes = sorted(list(scores.groups.keys()))

    # get the corresponding score
    score = scores.get_group(ws_sizes[0])

    # make the figure
    fig = make_signal_figure(signal, score, selected_signal)

    return fig, ws_sizes, ws_sizes[0]


@callback(
    Output('basic-interactions', 'figure', allow_duplicate=True),
    Input("session-id", "data"),
    Input("folder-name", "data"),
    Input('window-select', 'value'),
    State('signal-select', 'value'),
    prevent_initial_call=True
    )
def change_window_size(session_id: str, folder_name: str, select_window_size, selected_signal):
    # get the signal data and grouped score data
    _, _, _, _, _, raw_signals_grouped, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))
    scores_grouped, score_information = get_score_information(session_id, folder_name)

    # get the signal and score data
    signal = raw_signals_grouped.get_group(selected_signal)
    score = scores_grouped[selected_signal].get_group(select_window_size)

    return make_signal_figure(signal, score, selected_signal)


@callback(
    Output('hidden-graph2', 'figure'),
    Output('hidden-graph2-container', 'hidden', allow_duplicate=True),
    Input("session-id", "data"),
    Input("folder-name", "data"),
    Input('hidden-graph1', 'clickData'),
    State('signal-select', 'value'),
    State('basic-interactions', 'selectedData'),
    prevent_initial_call=True)
def display_click_data(session_id: str, folder_name: str, click_data, selected_signal, selected_data):
    # check whether we clicked a point
    if not click_data or not click_data['points'] or not selected_data or not selected_data['points']:
        return {}, True

    # get the minimum and maximum of all the selected points
    start = min(pd.Timestamp(point['x']) for point in selected_data['points'])
    end = max(pd.Timestamp(point['x']) for point in selected_data['points']) + pd.Timedelta(milliseconds=1)

    # get the signal data
    _, _, _, _, _, raw_signals_grouped, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # get the timezone of the selected signal
    tz = raw_signals_grouped.get_group(selected_signal).index.tz

    # set the timezone of the scores
    start = start.tz_localize(tz)
    end = end.tz_localize(tz)

    # get the name of the clicked signal
    cmpsig_name = click_data['points'][0]['customdata'][0]

    # make a figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # make the signal plot
    signal_df = raw_signals_grouped.get_group(selected_signal)
    fig.add_trace(
        go.Scatter(x=signal_df.index, y=signal_df['value'], name=selected_signal),
        secondary_y=False,
    )

    # make the score plot
    cmp_signal_df = raw_signals_grouped.get_group(cmpsig_name)
    fig.add_trace(
        go.Scatter(x=cmp_signal_df.index, y=cmp_signal_df['value'], name=cmpsig_name),
        secondary_y=True,
    )

    return fig, False


def layout(session_id: str, folder_name: str, **kwargs):

    # check whether we have a folder
    if not folder_name:
        return html.H1("Please upload a file using the sidebar.")

    # get some of the data to fill in the blanks and first selections
    scores, _, _, _, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))
    signal_names = list(scores.keys())

    layout_definition = html.Div([
        html.Div(children=[
            html.H1(f'Change Point Correlation Search',
                    style={'fontSize': 40},
                    id='header'),
        ],
            style=styles['div']
        ),
        html.Div(children=[
            dbc.Container(children=[
                dbc.Row(children=[
                    dbc.Col(children=[
                        'Signal Selection',
                        dcc.Dropdown(
                            id="signal-select",
                            options=signal_names,
                            value=signal_names[0],
                            multi=False,
                            clearable=False,
                            style=styles['dropdown'],
                            persistence="session",
                        ),
                    ],
                        style={"width": "100%"}
                    ),
                    dbc.Col(children=[
                        'Window Size',
                        dcc.Dropdown(
                            id="window-select",
                            options=[0, 1],
                            value=0,
                            multi=False,
                            clearable=False,
                            style=styles['dropdown'],
                            persistence="session",
                        ),
                    ],
                        style={"width": "100%"}
                    ),
                    dbc.Col(children=[
                        'Comparison Method',
                        dcc.Dropdown(
                            id="method-select",
                            options=["Pearson Correlation", "Spearman Correlation", "Activity"],
                            value="Pearson Correlation",
                            multi=False,
                            clearable=False,
                            style=styles['dropdown'],
                            persistence="session",
                        ),
                    ],
                        style={"width": "100%"}
                    ),
                ]),
            ],
                style={"width": "100%"},
            ),
            dcc.Graph(
                id='basic-interactions',
                figure={},
            ),
        ],
            style=styles['div']
        ),
        html.Div(children=[
            dcc.Graph(
                id='hidden-graph1',
                figure={}
            ),
            dcc.Dropdown(
                id="color-select",
                options=['Block', 'Turbine', 'Block+Turbine', 'Component'],
                value='Component',
                multi=False,
                clearable=False,
                style={key: val if key != 'width' else '50%' for key, val in styles['dropdown'].items()},
                persistence="session",
            )
        ],
            style=styles['div'],
            hidden=True,
            id='hidden-graph1-container'
        ),
        html.Div(children=[
            dcc.Graph(
                id='hidden-graph2',
                figure={}
            ),
        ],
            style=styles['div'],
            hidden=True,
            id='hidden-graph2-container'
        ),
        html.Div([
            dcc.Markdown("""
                        **Selection Data**

                        Choose the lasso or rectangle tool in the graph's menu
                        bar and then select points in the graph.

                    """),
            html.Pre(id='selected-data', style=styles['pre'])],
            style=styles['div'],
            hidden=True,
            id='selected-data-container'
        )
    ],
    )
    return layout_definition
