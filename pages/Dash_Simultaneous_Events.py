# inspired by https://dash.plotly.com/interactive-graphing
import logging
import time
import inspect

import plotly
from dash import dcc, html, Input, Output, callback, State, register_page, Patch
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import rankdata as scipyrank

import util.load_data as utl
import util.cache_registry as ucache
import util.styles as ustyles
import util.process_kks as ukks
import util.draw_heatmap as uheat
from GLOBALS import *

# register the page to our application
register_page(__name__, path="/correlate")
logger = logging.getLogger("frontend-logger")


@ucache.lru_cache(maxsize=CACHE_SIZE)
def get_score_information(session_id: str, folder_name: str):
    start = time.perf_counter()

    # get the scores from memory to preprocess them
    scores, _, _, _, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # get the minimum, maximum, mean and std of each score
    # new optimized way to get the information
    score_information = pd.concat(
        {
            name: grouped_df["value"].agg(["min", "max", "mean", "std"])
            for name, grouped_df in tqdm(scores.items(), desc='Compute the scoring statistics')
        },
        names=["name"],
    )
    score_information.index.names = ["name", "ws"]

    # return the result
    logger.info(f"[{__name__}] Preprocessed data in {time.perf_counter() - start:0.2f} s.")
    return score_information


@callback(
    Output('selected-data', 'children'),
    Output('hidden-graph1', 'figure'),
    Output('hidden-graph1-container', 'hidden'),
    Output('hidden-graph2-container', 'hidden'),
    Output('hidden-graph3-container', 'hidden'),
    Output('confirm-danger', 'displayed'),
    State("session-id", "data"),
    State("folder-name", "data"),
    Input('basic-interactions', 'selectedData'),
    Input('signal-select', 'value'),
    Input('color-select', 'value'),
    Input('window-select', 'value'),
    Input('method-select', 'value'),
    running=[
        (Output("correlate-events-basic-interactions-loading", "display"), "show", "auto"),
    ],
    prevent_initial_call=True,
)
def display_selected_data(session_id: str, folder_name: str,
                          selected_data, selected_signal, selected_color_choice, window_size, selection_method):

    # check whether there is something selected
    if not selected_data or not selected_data['points']:
        return "", {}, True, True, True, False

    # get the minimum and maximum of all the selected points
    min_idx = min(pd.Timestamp(point['x']) for point in selected_data['points'])
    max_idx = max(pd.Timestamp(point['x']) for point in selected_data['points']) + pd.Timedelta(milliseconds=1)

    # get the changepoint signal during this change
    cp_similarities, names = find_others(session_id, folder_name, min_idx, max_idx, selected_signal, window_size, selection_method)

    # check whether there is something selected
    if cp_similarities is None:
        return "", {}, True, True, True, True

    # construct the result string
    res_str = [f"Points from signal {selected_signal} are selected from Index: {min_idx} to {max_idx}.\n"]
    # res_str += [names[idx] for idx in res]

    # get the correlation values for the raw signals
    _, _, _, _, _, _, signal_correlation = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # get the correlation values for the selected signal
    corr_vals = signal_correlation.loc[names, selected_signal].abs().to_numpy()

    # extract the information from the names
    info_dict = ukks.get_info_from_list(names, unique=False)
    blocks = info_dict["block"]
    turbines = info_dict["turbine"]
    components = info_dict["component"]
    measurements = info_dict["measurement"]
    types = info_dict["type"]

    # make a figure
    # get_rand(len(cp_similarities))
    fig = px.scatter(pd.DataFrame({'Signal Corr.': corr_vals,
                                   'cp_similarity': cp_similarities,
                                   'name': names,
                                   'Block+Turbine': [f"{block}{turbine}" for block, turbine in zip(blocks, turbines)],
                                   'Block': blocks,
                                   'Component': components,
                                   'Measurement': measurements,
                                   'Turbine': [f'Steam [{turbine}]' if turbine == '0'
                                         else f'Gas [{turbine}]'
                                         for turbine in turbines]}),
                     x='cp_similarity',
                     y='Signal Corr.',
                     color=selected_color_choice,
                     hover_data=['cp_similarity', 'name'],
                     marginal_x="histogram",
                     title=f"CP Similarity for Signal {selected_signal} from index {min_idx} to {max_idx} ({max_idx-min_idx}).")

    # make the histogram stacked
    # fig.update_layout(barmode='group')
    fig.update_layout(transition_duration=250)
    fig.update_layout(xaxis_title="Change Point Similarity", yaxis_title="Complete Signal Correlation")

    # creates a gap between the scatter plot and the marginal
    fig.update_layout(
        yaxis=dict(domain=[0.00, 0.72]),  # main scatter
        yaxis2=dict(domain=[0.8, 1.00])  # top marginal
    )
    return "\n".join(res_str), fig, False, True, True, False


# make a python function that goes through all other signals and finds the most interesting ones
def find_others(session_id: str, folder_name: str, start, end, selected_signal, window_size, selection_method):
    _start = time.perf_counter()

    # get the data
    score_information = get_score_information(session_id, folder_name)
    scores_grouped, _, _, _, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))
    signal_names = list(scores_grouped.keys())

    # get the timezone of the selected signal
    tz = scores_grouped[selected_signal].get_group(window_size).index.tz

    # set the timezone of the scores
    start = start.tz_localize(tz)
    end = end.tz_localize(tz)

    # get the selected signal data
    signal = scores_grouped[selected_signal].get_group(window_size).loc[start:end, 'value'].to_numpy()

    # check whether we selected only one sample
    if signal.shape[0] < 1:
        return None, None

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
        signal = signal - signal_mean
        signal_std = np.sqrt(np.sum(np.square(signal)))

        # compute the mean of all signals
        signal_means = np.mean(other_array, axis=1)

        # subtract the mean from the array
        other_array = other_array - signal_means[:, None]

        # compute the standard deviations of the mean
        other_std = np.sqrt(np.sum(np.square(other_array), axis=1)) + np.finfo('float').eps

        # compute the covariance
        corr = (other_array @ signal[:, None])/(other_std * signal_std)[:, None]
        corr = corr[:, 0]

    elif selection_method == 'Activity':  # find the signal with the highest normalized activity

        # get the data for the min-max-normalization
        other_array_min = np.array([score_information.loc[(name, window_size), 'min'] for name in signal_names])
        other_array_max = np.array([score_information.loc[(name, window_size), 'max'] for name in signal_names])

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


def fuse_plotly_express_plots(fig1: plotly.graph_objects.Figure, fig2: plotly.graph_objects.Figure) -> plotly.graph_objects.Figure:
    """
    Plotly express does not allow secondary x-axis, but has the option for auto rendering.
    With this function, we get the auto rendering (mostly WebGL) and the secondary axis as the rendering option
    is chosen per trace.
    :param fig1: The figure for the first y-axis
    :param fig2: The figure for the second y-axis
    :return: Fused Figure
    """

    # make a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # get the color palette
    palette = list(fig1.layout.colorway) if fig1.layout.colorway else list(px.colors.qualitative.Plotly)
    offset = 0

    for tr in fig1.data:
        color = palette[offset % len(palette)]
        tr.update(
            line=dict(color=color),
            marker=dict(color=color),
        )
        offset += 1
        fig.add_trace(tr, secondary_y=False)

    for tr in fig2.data:
        color = palette[offset % len(palette)]
        tr.update(
            line=dict(color=color),
            marker=dict(color=color),
        )
        offset += 1
        fig.add_trace(tr, secondary_y=True)
    return fig


def make_signal_figure(signal_df, score_df, signal_name):
    fig1 = px.line(signal_df, y='value', markers=True, color='sensor')
    fig2 = px.line(score_df, y='value', color="window")
    fig2.update_traces(name='Change Score', showlegend=True)
    fig = fuse_plotly_express_plots(fig1, fig2)
    # update the layout
    fig.update_traces(marker_size=5)
    # fig.update_layout(transition_duration=500)
    fig.update_layout(dragmode='select')
    return fig


@callback(
    Output('basic-interactions', 'figure'),
    State("session-id", "data"),
    State("folder-name", "data"),
    Input('window-select', 'value'),
    Input('signal-select', 'value'),
    )
def change_window_size(session_id: str, folder_name: str, select_window_size, selected_signal):
    # get the signal data and grouped score data
    scores, _, _, _, _, raw_signals_grouped, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # get the signal and score data
    signal = raw_signals_grouped.get_group(selected_signal)
    score = scores[selected_signal].get_group(select_window_size)

    return make_signal_figure(signal, score, selected_signal)


@callback(
    Output('hidden-graph2', 'figure'),
    Output('hidden-graph2-container', 'hidden', allow_duplicate=True),
    Output('hidden-graph2-text', 'children'),
    State("session-id", "data"),
    State("folder-name", "data"),
    Input('hidden-graph1', 'clickData'),
    State('signal-select', 'value'),
    State('basic-interactions', 'selectedData'),
    State('window-select', 'value'),
    prevent_initial_call=True)
def display_signal_onclick(session_id: str, folder_name: str, click_data, selected_signal, selected_data, window_size):
    # check whether we clicked a point
    if not click_data or not click_data['points'] or not selected_data or not selected_data['points']:
        return {}, True, ""

    # get the minimum and maximum of all the selected points
    start = min(pd.Timestamp(point['x']) for point in selected_data['points'])
    end = max(pd.Timestamp(point['x']) for point in selected_data['points']) + pd.Timedelta(milliseconds=1)

    # get the signal data
    _, _, _, _, _, raw_signals_grouped, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # get the timezone of the selected signal
    tz = raw_signals_grouped.get_group(selected_signal).index.tz

    # set the timezone of the signals
    start = start.tz_localize(tz)
    end = end.tz_localize(tz)

    # get the name of the clicked signal
    cmpsig_name = click_data['points'][0]['customdata'][0]

    # make the signal plot
    signal_df = raw_signals_grouped.get_group(selected_signal)
    fig1 = px.line(signal_df, y='value', color='sensor')

    # make the other signal plot
    cmp_signal_df = raw_signals_grouped.get_group(cmpsig_name)
    fig2 = px.line(cmp_signal_df, y='value', color='sensor')

    # fuse the figures
    fig = fuse_plotly_express_plots(fig1, fig2)
    fig.update_layout(hovermode="x unified")

    # make the lines
    fig.add_vrect(x0=start, x1=end, annotation_text="selection")

    return fig, False, [html.H2('Signals'), f"Selected {selected_signal} & {cmpsig_name}"]


@callback(
    Output('hidden-graph3', 'figure'),
    Output('hidden-graph3-container', 'hidden', allow_duplicate=True),
    Output('hidden-graph3-text', 'children'),
    State("session-id", "data"),
    State("folder-name", "data"),
    Input('hidden-graph1', 'clickData'),
    State('signal-select', 'value'),
    State('basic-interactions', 'selectedData'),
    State('window-select', 'value'),
    prevent_initial_call=True)
def display_score_onclick(session_id: str, folder_name: str, click_data, selected_signal, selected_data, window_size):

    # check whether we clicked a point
    if not click_data or not click_data['points'] or not selected_data or not selected_data['points']:
        return {}, True, ""

    # get the minimum and maximum of all the selected points
    start = min(pd.Timestamp(point['x']) for point in selected_data['points'])
    end = max(pd.Timestamp(point['x']) for point in selected_data['points']) + pd.Timedelta(milliseconds=1)

    # get the signal data
    scores, _, _, _, _, raw_signals_grouped, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # get the timezone of the selected signal
    tz = raw_signals_grouped.get_group(selected_signal).index.tz

    # set the timezone of the scores
    start = start.tz_localize(tz)
    end = end.tz_localize(tz)

    # get the name of the clicked signal
    cmpsig_name = click_data['points'][0]['customdata'][0]

    # make the signal plot
    signal_df = scores[selected_signal].get_group(window_size)
    fig1 = px.line(signal_df, y='value', color="window")
    fig1.update_traces(name=selected_signal, showlegend=True)

    # make the other signal plot
    cmp_signal_df = scores[cmpsig_name].get_group(window_size)
    fig2 = px.line(cmp_signal_df, y='value', color="window")
    fig2.update_traces(name=cmpsig_name, showlegend=True)

    # fuse the figures
    fig = fuse_plotly_express_plots(fig1, fig2)
    fig.update_layout(hovermode="x unified")

    # make the lines
    fig.add_vrect(x0=start, x1=end, annotation_text="selection")

    return fig, False, [html.H2('Scores'), f"Selected {selected_signal} & {cmpsig_name}"]


@callback(
    Output(component_id='hidden-graph2', component_property="figure", allow_duplicate=True),
    Output(component_id='hidden-graph3', component_property="figure", allow_duplicate=True),
    Output(component_id='hidden-graph2', component_property="clickData"),
    Output(component_id='hidden-graph3', component_property="clickData"),
    Input(component_id='hidden-graph2', component_property="clickData"),
    Input(component_id='hidden-graph3', component_property="clickData"),
    prevent_initial_call=True)
def click_in_residuals(click_data1, click_data2):

    # check whether we selected nothing
    if click_data1 is None and click_data2 is None:
        raise PreventUpdate

    # get the signals we selected
    if click_data1 is not None:
        selected_time = [point['x'] for point in click_data1["points"]][0]
    else:
        selected_time = [point['x'] for point in click_data2["points"]][0]

    # make some patches
    figure_patch = Patch()
    del figure_patch['layout']['shapes'][1]
    figure_patch['layout']['shapes'].append(uheat.make_vline(selected_time))
    return figure_patch, figure_patch, None, None


def layout(session_id: str, folder_name: str, **kwargs):

    # get the start time
    start = time.perf_counter()

    # log the request
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Requested the simultaneous events page.")

    # check whether we have a folder
    if not folder_name:
        return html.H1("Please upload a file using the sidebar.")

    # check whether we have all necessary data
    scores, _, window_sizes, _, _, _, raw_signal_correlations = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))
    if raw_signal_correlations is None:
        return html.H1("Your file does not contain the 'signal_correlation.parquet' file.")

    # preprocess the score information
    get_score_information(session_id, folder_name)

    # get some of the data to fill in the blanks and first selections
    signal_names = list(scores.keys())

    layout_definition = html.Div([
        dcc.ConfirmDialog(
            id='confirm-danger',
            message='Please Select more than one data point!',
        ),
        html.Div(children=[
            html.H1(f'Change Point Correlation Search',
                    style={'fontSize': 40},
                    id='header'),
        ],
            style=ustyles.div_styles['div']
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
                            style=ustyles.div_styles['dropdown'] | {"width": "100%"},
                        ),
                    ],
                        style={"width": "100%"}
                    ),
                    dbc.Col(children=[
                        'Window Size',
                        dcc.Dropdown(
                            id="window-select",
                            options=window_sizes,
                            value=min(window_sizes),
                            multi=False,
                            clearable=False,
                            style=ustyles.div_styles['dropdown'] | {"width": "100%"},
                        ),
                    ],
                        style={"width": "100%"}
                    ),
                    dbc.Col(children=[
                        'Comparison Method',
                        dcc.Dropdown(
                            id="method-select",
                            options=["Pearson Correlation", "Spearman Correlation", "Activity"],
                            value="Spearman Correlation",
                            multi=False,
                            clearable=False,
                            style=ustyles.div_styles['dropdown'] | {"width": "100%"},
                        ),
                    ],
                        style={"width": "100%"}
                    ),
                ]),
            ],
                style={"width": "100%"},
            ),
            dcc.Loading(children=[
                dcc.Graph(
                    id='basic-interactions',
                    figure={},
                ),
            ],
                type="circle",
                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                id="correlate-events-basic-interactions-loading",
            ),
        ],
            style=ustyles.div_styles['div']
        ),
        html.Div(children=[
            dcc.Loading(children=[
                dcc.Graph(
                    id='hidden-graph1',
                    figure={}
                ),
            ],
                type="circle",
                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                id="correlate-events-scatter-signals-loading",
            ),
            dcc.Dropdown(
                id="color-select",
                options=['Block', 'Turbine', 'Block+Turbine', 'Component'],
                value='Component',
                multi=False,
                clearable=False,
                style=ustyles.div_styles['dropdown'],
            )
        ],
            style=ustyles.div_styles['div'],
            hidden=True,
            id='hidden-graph1-container'
        ),
        html.Div(children=[
            html.Div(children=[], id='hidden-graph2-text', style={"text-align": "left"}),
            dcc.Graph(
                id='hidden-graph2',
                figure={}
            ),
        ],
            style=ustyles.div_styles['div'],
            hidden=True,
            id='hidden-graph2-container'
        ),
        html.Div(children=[
            html.Div(children=[], id='hidden-graph3-text', style={"text-align": "left"}),
            dcc.Graph(
                id='hidden-graph3',
                figure={}
            ),
        ],
            style=ustyles.div_styles['div'],
            hidden=True,
            id='hidden-graph3-container'
        ),
        html.Div([
            dcc.Markdown("""
                        **Selection Data**

                        Choose the lasso or rectangle tool in the graph's menu
                        bar and then select points in the graph.

                    """),
            html.Pre(id='selected-data', style=ustyles.div_styles['pre'])],
            style=ustyles.div_styles['div'],
            hidden=True,
            id='selected-data-container'
        )
    ],
    )

    # log the duration
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Created the simultaneous events page in {time.perf_counter() - start:0.2f} s.")

    return layout_definition

if __name__ == '__main__':
    get_score_information("download_zip", "2025-09-10T143441Z_df_raw_lucas")