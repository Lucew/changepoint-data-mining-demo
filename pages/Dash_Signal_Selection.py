import logging
import time
from time import perf_counter
import re
import inspect

import dash
from dash import dcc, html, Input, Patch, Output, callback, State, clientside_callback, ctx, register_page, ClientsideFunction, ALL, stringify_id
from dash_extensions import EventListener
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

import util.load_data as utl
import util.prepocessing as prep
from GLOBALS import *
import util.residuals as procd
import util.cache_registry as ucache


# register the page to our application
register_page(__name__, path="/scatter")
logger = logging.getLogger("frontend-logger")

# set an id for the score graph
score_graph_type = "raw-signal-graph"
score_graph_index = "scatter-signal-graph"
score_graph_id = {"type": score_graph_type, "index": score_graph_index}


# define some styles for different html elements
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'color': 'black'
    },
    'div': {
        'padding': '.6rem',
        'width': '90%',
        'margin': 'auto',
        'boxShadow': 'dimgrey 4px 4px 2px',
        'borderRadius': '10px',
        'backgroundColor': 'white',
        'marginTop': '1rem',
        'resize': 'vertical',
        'overflowX': 'hidden',
        'overflowY': 'auto',
    },
    'stickydiv': {
        'position': 'sticky',
        'top': '-10px',
        "zIndex": "20",
        'overflowX': 'clip',
        'overflowY': 'clip',
    },
    'dropdown': {
        'margin': 'auto',
        'width': '50%',
        'borderRadius': '10px',
        'color': 'black'
    },
}


def draw_heatmap(data: pd.DataFrame):
    # normalize the data
    data = prep.normalization(data)

    # make the figure
    fig = px.imshow(data.transpose())

    # Define dragmode, newshape parameters, amd add modebar buttons
    fig.update_layout(
        dragmode='drawrect',  # define dragmode
        newshape=dict(line_color='cyan'),
        modebar_add=[
            'drawrect',
        ],
    )
    fig.update_xaxes(scaleanchor=False)
    fig.update_layout(uirevision="keep")

    return fig


@ucache.cache
def get_random_state():
    # return np.random.RandomState(25)
    return np.random.RandomState(3)


@ucache.lru_cache(maxsize=1)
def filter_regression_results(session_id: str, folder_name: str,
                              correlation_threshold: float = None) -> (pd.DataFrame, pd.DataFrame, float):
    start = time.perf_counter()
    # get the preprocessed regression results
    regression_results, _, complete_max_correlation = prep.preprocess_regression_results(session_id, folder_name)

    # create default correlation threshold if not given
    if correlation_threshold is None:
        correlation_threshold = 0.90

    # find all the signal names that have larger correlation than the threshold
    mask = complete_max_correlation["correlation"] > correlation_threshold
    max_correlated_signal_tags = set(complete_max_correlation[mask].index)

    # only keep the signals that have maximum correlation
    filtered_regression_results = regression_results.loc[regression_results["x"].isin(max_correlated_signal_tags) &
                                                         regression_results["y"].isin(max_correlated_signal_tags)]

    # fill the distance matrix for the signals using pivoting of the regression results
    # https://stackoverflow.com/questions/71671054/convert-pandas-dataframe-to-distance-matrix
    filtered_distance_matrix = filtered_regression_results.pivot(index='x', columns='y', values='distance').fillna(0)

    # return the filtered regression results
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Filtered regression results in {time.perf_counter() - start:0.2f} s.")
    return filtered_regression_results, filtered_distance_matrix, correlation_threshold


@ucache.lru_cache(maxsize=1)
def create_tsne(session_id: str, folder_name: str,
                perplexity: int = None, correlation_threshold: float = None) -> (pd.DataFrame, int, float):
    # TODO: Recompute TSNE when different signals are selected

    # make the timing
    started = perf_counter()

    # get the filtered regression results
    _, distance_matrix, correlation_threshold = filter_regression_results(session_id, folder_name, correlation_threshold)

    # load the raw data
    _, _, _, anomaly_scores, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # load the maximum threshold information and signal names
    _, _, max_correlation = prep.preprocess_regression_results(session_id, folder_name)

    # compute default perplexity
    # print("Parameters", session_id, folder_name, perplexity, correlation_threshold)
    if perplexity is None:
        perplexity = int(np.sqrt(distance_matrix.shape[0]))

    # get the random state
    random_state = get_random_state()

    # compute the TSNE representation of the data
    tsne = TSNE(metric='precomputed', init='random', perplexity=perplexity, random_state=random_state)
    transf = tsne.fit_transform(distance_matrix)

    # compute the TSNE representation of the data
    tsne3d = TSNE(n_components=3, metric='precomputed', init='random', perplexity=perplexity, random_state=random_state)
    transf3d = tsne3d.fit_transform(distance_matrix)

    # create the dataframe from the information
    # make the dataframe that we want to have
    bokeh_df = pd.DataFrame({'x': transf[:, 0],
                             'y': transf[:, 1],
                             'x3d': transf3d[:, 0],
                             'y3d': transf3d[:, 1],
                             'z3d': transf3d[:, 2],
                             'original': list(distance_matrix.columns),
                             'Max. Corr.': [f"{ele:0.3f} " for ele in
                                            list(max_correlation.loc[distance_matrix.columns]["correlation"])],
                             'Anomaly Score': anomaly_scores.loc[distance_matrix.columns, 'score'] if anomaly_scores is not None else [None]*transf.shape[0],
                             'block': [col[1:2] for col in list(distance_matrix.columns)],
                             'block_turbine': [col[1:3] for col in list(distance_matrix.columns)],
                             'turbine': [f'Steam [{col[2:3]}]' if col[2:3] == '0'
                                         else f'Gas [{col[2:3]}]'
                                         for col in list(distance_matrix.columns)],
                             'component': [col[3:6] for col in list(distance_matrix.columns)],
                             'measurement': [col[8:10] for col in list(distance_matrix.columns)]
                             })

    # TODO TSNE perplexity as component group size?
    # print((bokeh_df.groupby("component")["x"].count()).median())

    # recompute the anomaly scores
    rec_anomaly = [0]*transf.shape[0]
    if anomaly_scores is not None:
        rec_anomaly = bokeh_df.loc[:, 'Anomaly Score'].to_numpy().copy()
        rec_anomaly[np.isnan(rec_anomaly)] = 0
        zeroless_mask = rec_anomaly != 0
        zero_mask = ~zeroless_mask
        rec_zeroless = np.log(rec_anomaly[zeroless_mask])
        rec_anomaly[zero_mask] = rec_zeroless.min()
        rec_anomaly[zeroless_mask] = rec_zeroless
        rec_anomaly = (rec_anomaly - rec_anomaly.min()) / (rec_anomaly.max() - rec_anomaly.min())
    bokeh_df['opac'] = rec_anomaly

    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Created TSNE ({perf_counter()-started:0.2f} s).")
    return bokeh_df, perplexity, correlation_threshold


def prepare_plot_data(session_id: str, folder_name: str, perplexity: int = None, correlation_threshold: int = None,
                      selected_components: list[str] = None, selected_measurements: list[str] = None):
    # load the data
    bokeh_df, _, _ = create_tsne(session_id, folder_name, perplexity, correlation_threshold)

    # check whether we need to select something
    if selected_components is not None:

        # select the data we want to have
        selected_components = set(selected_components)
        bokeh_df = bokeh_df[bokeh_df['component'].isin(selected_components)]
    if selected_measurements is not None:

        # select the data we want to have
        selected_measurements = set(selected_measurements)
        bokeh_df = bokeh_df[bokeh_df['measurement'].isin(selected_measurements)]
    # print("left", bokeh_df[["original", "x", "y"]])
    return bokeh_df


def create_scatter(session_id: str, folder_name: str, perplexity: int = None, correlation_threshold: float = None,
                   selected_components: list[str] = None, selected_measurements: list[str] = None):

    # get the scatter plot information
    bokeh_df = prepare_plot_data(session_id, folder_name, perplexity, correlation_threshold, selected_components, selected_measurements)

    # make the figure
    fig = px.scatter(bokeh_df, x="x", y="y", color="component", symbol="measurement", size="opac",
                     hover_data=["original", 'Max. Corr.', 'Anomaly Score'])
    fig.update_traces(marker=dict(
                                  line=dict(width=2,
                                            color='DarkSlateGrey',
                                            ),
                                  ),
                      selector=dict(mode='markers'))

    # go through the markers and change their color
    # https://stackoverflow.com/a/68175130
    for data in fig.data:
        data['marker']['line'].width = [opac*5 for opac in data['marker']['size']]
        data['marker'].size = 12

    # set the selection style
    fig.update_layout(
        dragmode='select',  # define from: ['zoom', 'pan', 'select', 'lasso', 'drawclosedpath', 'drawopenpath', 'drawline', 'drawrect', 'drawcircle', 'orbit', 'turntable', False]
    )
    fig.update_layout(
        autosize=True,
    )
    return fig


def create_scatter_3d(session_id: str, folder_name: str,
                      perplexity: int = None, correlation_threshold: float = None,
                      selected_components: list[str] = None, selected_measurements: list[str] = None):

    # get the scatter plot information
    bokeh_df = prepare_plot_data(session_id, folder_name, perplexity, correlation_threshold, selected_components, selected_measurements)

    # make the figure
    # NOTE: Linking the 3D plot to the 2D is not trivial as selectedpoints is not a valid option
    # for scatter_3d as of now
    fig = px.scatter_3d(bokeh_df, x="x3d", y="y3d", z='z3d', color="component", symbol="measurement",
                        hover_data=["original"])

    return fig


def find_nearest_index(df: pd.DataFrame, time_start, time_end) -> (int, int):
    start_idx, end_idx = sorted(df.index.get_indexer([pd.Timestamp(time_start), pd.Timestamp(time_end)],
                                                     method='nearest'))
    return start_idx, end_idx


def make_histogram(session_id: str, folder_name: str, correlation_threshold: float):

    # load the regression results
    _, _, complete_max_correlation = prep.preprocess_regression_results(session_id, folder_name)

    # make the histogram out of it
    fig = px.ecdf(complete_max_correlation, x="correlation", marginal="histogram", ecdfnorm=None)
    mask = complete_max_correlation["correlation"] > correlation_threshold
    fig.add_vline(correlation_threshold, annotation_text=f'Remaining Tags: {mask.sum()}')
    return fig


# app callbacks --------------------------------------------------------------------------------------------------------

@callback(
    Output(score_graph_id, component_property="figure", allow_duplicate=True),
    Output(component_id='div-scatter-signal-graph', component_property="hidden"),
    Output(component_id='scatter-overall-div', component_property='hidden'),
    Output(component_id='scatter-overall-div2', component_property='hidden', allow_duplicate=True),
    Output(component_id='div-scatter-signal-graph2', component_property='hidden', allow_duplicate=True),
    Output("graph-loading-signal", "children", allow_duplicate=True),
    Output("signal-name-store", "data"),
    State("session-id", "data"),
    State("folder-name", "data"),
    Input(component_id="scatter-graph", component_property="selectedData"),
    prevent_initial_call=True)
def select_signals_scatter(session_id: str, folder_name: str, selected_data):

    # check whether we selected nothing
    if selected_data is None or len(selected_data["points"]) == 0:
        return go.Figure(go.Scatter(x=[], y=[])), True, True, dash.no_update, dash.no_update, "", []

    # get the signals we selected
    selected_signals = [point["customdata"][0] for point in selected_data["points"]]

    # get the regression results
    regression_results, _, _ = prep.preprocess_regression_results(session_id, folder_name)

    # get the scores from the files
    scores, _, _, _, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # compute the weighted scoring for each of the selected signals
    # TODO mean correlation value per cluster as hint whether it is a good cluster
    result_df = procd.compute_weighted_residual_norm(regression_results, selected_signals, scores, coming_from='signal-selection')

    # make the figure from the signals
    fig = draw_heatmap(result_df)

    return fig, False, False, True, True, "", list(result_df.columns)


@callback(
    Output(component_id="scatter-signal-graph2", component_property="figure", allow_duplicate=True),
    Output(component_id="raw-scatter-signal-graph", component_property="figure", allow_duplicate=True),
    Output(component_id='scatter-overall-div2', component_property='hidden'),
    Output(component_id='div-scatter-signal-graph2', component_property='hidden'),
    Output("graph-loading-signal", "children", allow_duplicate=True),
    State("session-id", "data"),
    State("folder-name", "data"),
    Input(component_id="scatter-graph", component_property="clickData"),
    prevent_initial_call=True)
def click_signals_scatter(session_id: str, folder_name: str, click_data):

    # check whether we selected nothing
    if click_data is None:
        return go.Figure(go.Scatter(x=[], y=[])), go.Figure(go.Scatter(x=[], y=[])), True, True, ""

    # get the signals we selected
    selected_signal = [point["customdata"][0] for point in click_data["points"]][0]

    # get the regression results
    _, extended_regression_results_grouped, _ = prep.preprocess_regression_results(session_id, folder_name)

    # get the score data from the files
    scores, _, _, _, _, raw_signals, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # get the regression results of the current signal
    regression_results = extended_regression_results_grouped.get_group(selected_signal)

    # get the largest neighbors
    neighbors = regression_results.nlargest(5, "correlation")["y"].to_list()
    neighbors.append(selected_signal)
    result_df = procd.compute_weighted_residual_norm(regression_results, neighbors, scores, coming_from='signal-selection2')

    # load the raw signals
    raw_signals = pd.concat(raw_signals.get_group(grp) for grp in neighbors)

    # make a colormap for both figures
    # https://stackoverflow.com/a/68209157
    c = dict(zip(neighbors, px.colors.qualitative.G10))

    # make the figure from the signals
    fig = px.line(result_df, color_discrete_map=c)
    fig2 = px.line(raw_signals, y="normalized value", color='sensor', color_discrete_map=c, hover_data="value")

    # reorder the lines so the sensor itself is plotted last and the neighbors are in order
    sort_keys = {name: idx for idx, name in enumerate(neighbors)}
    fig2.data = sorted(fig2.data, key=lambda x: sort_keys[x['name']])
    fig.data = sorted(fig.data, key=lambda x: sort_keys[x['name']])

    # customize the legend and labels
    fig.update_layout(legend_title='residuals')
    fig.update_layout(xaxis_title="Date", yaxis_title="Residual Value (Unexpected Change)")
    fig2.update_layout(xaxis_title="Date", yaxis_title="Normalized Signal Value")

    return fig, fig2, False, False, ""


def make_vline(x_position: float, width: float = 2, color: str = "black", dash_type: str = "solid"):

    line = {
        "type": "line",
        "xref": "x",
        "yref": "paper",  # span full plotting height
        "x0": x_position,
        "x1": x_position,
        "y0": 0,
        "y1": 1,
        "line": {
            "color": color,
            "width": width,
            "dash": dash_type,
        },
    }
    return line


@callback(
    Output(component_id="scatter-signal-graph2", component_property="figure", allow_duplicate=True),
    Output(component_id="raw-scatter-signal-graph", component_property="figure", allow_duplicate=True),
    Output(component_id="raw-scatter-signal-graph", component_property="clickData"),
    Output(component_id="scatter-signal-graph2", component_property="clickData"),
    Input(component_id="raw-scatter-signal-graph", component_property="clickData"),
    Input(component_id="scatter-signal-graph2", component_property="clickData"),
    prevent_initial_call=True)
def click_in_residuals(click_data1, click_data2):
    print(click_data1, click_data2)
    # check whether we selected nothing
    if click_data1 is None and click_data2 is None:
        raise PreventUpdate

    # get the signals we selected
    if click_data1 is not None:
        selected_time = [point['x'] for point in click_data1["points"]][0]
    else:
        selected_time = [point['x'] for point in click_data2["points"]][0]
    print(selected_time)
    # make some patches
    figure_patch = Patch()
    figure_patch['layout']['shapes'].clear()
    figure_patch['layout']['shapes'].append(make_vline(selected_time))
    return figure_patch, figure_patch, None, None


@callback(
    Output('scatter-graph', 'figure'),
    Output('scatter-graph3d', 'figure'),
    State("session-id", "data"),
    State("folder-name", "data"),
    Input('perplexity-slider', 'value'),
    Input('component-select', 'value'),
    Input('measurement-select', 'value'),
    Input('correlation-slider', 'value'),
    prevent_initial_call=True)
def update_scatter_plots(session_id: str, folder_name: str,
                         value, component_list, measurement_list, correlation_threshold):
    return (create_scatter(session_id, folder_name, value, correlation_threshold, component_list, measurement_list),
            create_scatter_3d(session_id, folder_name,value, correlation_threshold, component_list, measurement_list))


@callback(
    Output('histogram-graph', 'figure'),
    Output('corr-val-div', 'children'),
    Input('correlation-slider', 'value')
)
def update_histogram(correlation_threshold: float):
    histogram_figure_patch = Patch()
    histogram_figure_patch['layout']['shapes'].clear()
    histogram_figure_patch["layout"]["shapes"].append(make_vline(correlation_threshold))
    return histogram_figure_patch, f'Correlation Value: {correlation_threshold}'


def make_selection_title(idx: int):
    return f'Selection {idx}'

def make_raw_signal_plot_title(idx: int):
    return f'Signal for {make_selection_title(idx)}'

def is_custom_shape(shape: dict):
    return shape['type'] == 'rect' or shape['type'] == 'path'

@callback(
    Output(component_id=score_graph_id, component_property='figure', allow_duplicate=True),
    Output(component_id="scatter-delete-listener", component_property="event", allow_duplicate=True),
    Output(component_id="scatter-overall-div", component_property="children"),
    Input(component_id="scatter-delete-listener", component_property="n_events"),
    State(component_id="scatter-delete-listener", component_property="event"),
    prevent_initial_call=True
)
def delete_shapes(n_delete_events, delete_event_data):

    if delete_event_data is None:
        raise PreventUpdate

    # check whether there are existing rectangle shapes
    shape_idces = [idx for idx, ele in enumerate(delete_event_data['detail.shapes']) if is_custom_shape(ele)]
    if not shape_idces:
        raise PreventUpdate

    # write to logger
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {ctx.triggered_id}, {ctx.inputs if len(str(ctx.inputs)) < 400 else f'{str(ctx.inputs)[:250]} [...]'}")

    # create our patch objects for our figure
    figure_shape_patch = Patch()
    scatter_overall_patch = Patch()

    # get the index of the active shape
    active_idx = delete_event_data['detail.children']

    # delete the shape
    if not active_idx:
        active_idx = [shape_idces[-1]]
    elif active_idx == 'all':
        active_idx = shape_idces
    else:
        active_idx = list(map(int, active_idx))

    active_idx = set(active_idx)

    # update the newest shape accordingly
    new_index = 1
    for idx in shape_idces:
        if idx in active_idx:
            continue

        # adapt the text of the selections
        figure_shape_patch['layout']['shapes'][idx]['label']['text'] = make_selection_title(new_index)

        # update the corresponding raw signal plot headers
        scatter_overall_patch[idx]["props"]["children"][0]["props"]["children"] = make_raw_signal_plot_title(new_index)

        # increment the new index
        new_index += 1

    # delete the shapes in the figure
    # !IMPORTANT! Do this as last step. Otherwise, shape indices will be incorrect.
    # !IMPORTANT! Do this in reverse index order
    for ele in sorted(active_idx, reverse=True):
        del figure_shape_patch['layout']['shapes'][ele]
        del scatter_overall_patch[ele]

    return figure_shape_patch, None, scatter_overall_patch


def shape_update_patch(shape: dict, figure_shape_patch: Patch, shape_idx: int, title_idx: int = None):

    # get the indices depending on the type of shape
    if shape['type'] == 'rect':
        shape_y0 = shape['y0']
        shape_y1 = shape['y1']
    elif shape['type'] == 'path':
        parts = re.findall(r'([ML])([^ML]+)', shape['path'])
        shape_y0 = min(float(ele[1].split(',', 1)[1]) for ele in parts)
        shape_y1 = max(float(ele[1].split(',', 1)[1]) for ele in parts)
        shape['x0'] = min(ele[1].split(',', 1)[0].replace('_', ' ') for ele in parts)
        shape['x1'] = max(ele[1].split(',', 1)[0].replace('_', ' ') for ele in parts)
    else:
        raise ValueError(f'Shape of type {shape['type']} not defined.')

    # round the indices
    shape_y0 = round(max(shape_y0, 0))
    shape_y1 = round(shape_y1)

    # figure_shape_patch['layout']['shapes'][shape_idx]['y0'] = shape_y0
    # figure_shape_patch['layout']['shapes'][shape_idx]['y1'] = shape_y1

    # give the shape a text if it is a new shape
    if title_idx is not None:
        figure_shape_patch['layout']['shapes'][shape_idx]['label']['text'] = make_selection_title(title_idx)
        figure_shape_patch['layout']['shapes'][shape_idx]['label']['font'] = {'color': 'white'}

    return shape_y0, shape_y1


def create_raw_signal_figure(session_id: str, folder_name: str, shape: dict, shape_y0: int, shape_y1: int, signal_names: list[str]):
    # load the raw signals
    _, _, _, _, _, signals, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # get all the columns we need to select (depending on where we start, one of the shapes is the larger one)
    column_start, column_end = sorted((shape_y0, shape_y1))

    # get the time we need to select (depending on where we start, one of the shapes is the larger one)
    time_start = pd.Timestamp(shape['x0'])
    time_end = pd.Timestamp(shape['x1'])
    if time_end < time_start:
        time_start, time_end = time_end, time_start

    # get the signal names from the column indices
    names = signal_names[column_start:column_end + 1]

    # get the signal data
    names = set(names)
    tmp_signals = pd.concat(signals.get_group(grp).loc[time_start:time_end] for grp in names)

    # make a figure using the first signal
    fig = px.line(tmp_signals, y='normalized value', line_group='sensor', color='sensor', custom_data='value',
                  hover_data=['sensor', 'value'], markers=True)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(
        uirevision="keep-zoom",
    )

    return fig, names


@callback(
Output(component_id=score_graph_id, component_property='figure', allow_duplicate=True),
    Output(component_id='scatter-overall-div', component_property='children', allow_duplicate=True),
    State("session-id", "data"),
    State("folder-name", "data"),
    State("signal-name-store", "data"),
    Input(component_id=score_graph_id, component_property='relayoutData'),
    prevent_initial_call=True
)
def create_new_raw_signal_plot(session_id: str, folder_name: str, signal_names: list[str], relayout_data: dict):

    # check whether the relayoutData was the trigger, but it was only sizing
    is_new_shape = 'shapes' in relayout_data
    if not is_new_shape:
        raise PreventUpdate

    # get existing rectangle shapes
    shapes = [(idx, ele) for idx, ele in enumerate(relayout_data.get('shapes', [])) if is_custom_shape(ele)]

    # if there are no rectangle shapes we do not have to do anything
    if not shapes:
        raise PreventUpdate

    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {ctx.triggered_id}, {ctx.inputs if len(str(ctx.inputs)) < 400 else f'{str(ctx.inputs)[:250]} [...]'}")

    # create our patch objects for the raw signal plots
    signal_plot_patch = Patch()
    figure_shape_patch = Patch()

    # get the latest shape and its number
    if is_new_shape:
        idx = len(shapes)
        shape_dx, shape = shapes[idx-1]
    else:
        logger.error(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {ctx.triggered_id}. We did not expect {relayout_data=} to get this far.")
        raise PreventUpdate

    # update the shape
    shape_y0, shape_y1 = shape_update_patch(shape, figure_shape_patch, shape_dx, idx)

    # get the current time as an into to grant unique ids
    currtime = time.time_ns()

    # create the figure
    if len(shapes) > MAX_PLOTLY_SHAPES-3:
        raw_signal_graph = html.Div([html.A("Too many shapes", href='https://plotly.com/python/performance/'), ". Please delete some. Otherwise, rendering will fail."])
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Too many shapes: {len(shapes)=}.")
    else:
        fig, names = create_raw_signal_figure(session_id, folder_name, shape, shape_y0, shape_y1, signal_names)
        raw_signal_graph = dcc.Graph(figure=fig, id={"type": "raw-signal-graph", "index": currtime})

    # create the new div
    new_raw_plot = html.Div(children=[html.H3(make_raw_signal_plot_title(idx)),
                                      html.Details(children=[
                                          dcc.Loading(children=[
                                              html.Div(children=[
                                                  raw_signal_graph,
                                              ],
                                                  id={"type": "raw-signal-graph-div", "index": currtime},
                                              )
                                          ],
                                              overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                                          )
                                      ],
                                          open=True,
                                          id={"type": "raw-signal-div", "index": currtime},
                                      )],
                            style=styles['div'],
                            id=f'scatter-signal-selection-div-{currtime}',
                            )

    # append the div to the existing divs
    signal_plot_patch.append(new_raw_plot)

    return figure_shape_patch, signal_plot_patch


@callback(
Output({"type": "raw-signal-graph-div", "index": ALL}, "children"),
    Output(component_id="scatter-move-listener", component_property="event", allow_duplicate=True),
    Output(component_id=score_graph_id, component_property='figure', allow_duplicate=True),
    State("session-id", "data"),
    State("folder-name", "data"),
    State("signal-name-store", "data"),
    Input(component_id="scatter-move-listener", component_property="n_events"),
    State(component_id="scatter-move-listener", component_property="event"),
    running=[(Output("loading-signals", "display"), "show", "auto")],
    prevent_initial_call=True,
)
def move_score_shape(session_id: str, folder_name: str, signal_names: list[str], n_events: int, event_data: dict):

    if event_data is None:
        raise PreventUpdate

    # write to logger
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {ctx.triggered_id}, {ctx.inputs if len(str(ctx.inputs)) < 400 else f'{str(ctx.inputs)[:250]} [...]'}")

    # create our patch objects for our figure
    figure_shape_patch = Patch()
    graph_patch = Patch()

    # get the index of the active shape
    all_shapes = event_data['detail.children']
    relayout_data = event_data['detail.relayout_data']

    # extract the current shape information
    pattern = re.compile(r'\[(\d+)]\.(\w+)$')
    elements_dict = {(int(n), term): value for (n, term), value in
                     ((re.search(pattern, key).groups(), value) for key, value in relayout_data.items())}

    # check that the shape index is unique
    shape_index = set(ele[0] for ele in elements_dict.keys())

    # check our extraction results
    if len(shape_index) != 1:
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {ctx.triggered_id}. We did not find relocate variables in {relayout_data=}.")
        raise PreventUpdate

    # get the index
    shape_dx = shape_index.pop()

    # get existing rectangle shapes
    shapes = [idx for idx, ele in enumerate(all_shapes) if is_custom_shape(ele)]

    # find the div idx
    div_dx = shapes.index(shape_dx)
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Moved shape {shape_dx} with corresponding div {div_dx}.")

    # check whether we have too many shapes
    if len(shapes) > MAX_PLOTLY_SHAPES-3 and shape_dx >= shapes[MAX_PLOTLY_SHAPES-3]:
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Too many shapes: {len(shapes)=}.")
        raise PreventUpdate

    # build a dummy shape with all the necessary information
    shape = {key[1]: val for key, val in elements_dict.items()}
    shape['type'] = all_shapes[shape_dx]['type']

    # update the shape
    shape_y0, shape_y1 = shape_update_patch(shape, figure_shape_patch, shape_dx, None)

    # get the current time as an into to grant unique ids
    currtime = time.time_ns()

    # create the figure
    fig, _ = create_raw_signal_figure(session_id, folder_name, shape, shape_y0, shape_y1, signal_names)
    graph = dcc.Graph(figure=fig, id={"type": "raw-signal-graph", "index": currtime})

    # create the update list
    graph_patch[0] = graph
    update_list = [graph_patch if idx == div_dx else dash.no_update for idx in range(len(shapes))]

    return update_list, None, figure_shape_patch


@callback(
    Output("shape-event-dummy", "children", allow_duplicate=True),
    Input("shape-event-store", "data"),
    prevent_initial_call=True,
)
def handle_shape_event(evt):
    if not evt:
        raise PreventUpdate

    # evt["type"] in {"newShape", "shapeRedraw", "deleteShape", ...}
    print(evt)
    raise PreventUpdate
    return f"Got {evt['type']}"


# how to call an update from button presses modified from
# https://community.plotly.com/t/how-to-get-trigger-from-keyboard/76960/6
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="delete_stuff_scatter"),
    Output(score_graph_id, "id"),
    Input("div-scatter-signal-graph", "id"),
    State('scatter-overall-div', "id"),
)

# delete all shapes
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="delete_all_stuff"),
    Output("scatter-delete-button-all", "id"),
    Input("scatter-delete-button-all", "id"),
    Input("div-scatter-signal-graph", "id")
)

# delete active shapes
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="delete_active_shape"),
    Output("scatter-delete-button-active", "id"),
    Input("scatter-delete-button-active", "id"),
    Input("div-scatter-signal-graph", "id")
)

# frontend create line shapes on click within raw signal plots
clientside_callback(
    ClientsideFunction(namespace="clientside3", function_name="sync_raw_signal_hover"),
Output({"type": "raw-signal-graph", "index": ALL}, "figure", allow_duplicate=True),
    Output("raw-signal-selected-x", "data"),
    Input({"type": "raw-signal-graph", "index": ALL}, "clickData"),
    State({"type": "raw-signal-graph", "index": ALL}, "id"),
    State({"type": "raw-signal-graph", "index": ALL}, "relayoutData"),
    State({"type": "raw-signal-graph", "index": ALL}, "figure"),
    State("raw-signal-selected-x", "data"),
    prevent_initial_call=True,
)

# function to emit events when shapes are moved
clientside_callback(
    ClientsideFunction(namespace="move_shapes_namespace", function_name="move_shapes"),
Output('correlation-slider', "id"),
    Input(score_graph_id, "relayoutData"),
    State("div-scatter-signal-graph", "id"),
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="shape_events", function_name="handle_relayout"),
    Output("shape-event-dummy", "children", allow_duplicate=True),
    Input(score_graph_id, "relayoutData"),
    State(score_graph_id, "figure"),
    State("shape-event-store", "id"),   # pass target store id into JS
    prevent_initial_call=True,
)


# dash app layout ------------------------------------------------------------------------------------------------------
def layout(session_id: str, folder_name: str, **kwargs):

    # check whether we have a folder
    if not folder_name:
        return html.H1("Please upload a file using the sidebar.")

    # check whether we have all necessary data
    _, _, _, anomaly_scores, distances, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))
    if distances is None:
        return html.H1("Your zip-file does not contain the anomaly_scores.parquet or/and the distances.csv file(s).")

    # define an event for our custom event listener to drag and drop buttons
    delete_event = {"event": 'shapeDeletion', "props": ["detail.children", "detail.shapes"]}
    move_event = {"event": 'shapeMove', "props": ["detail.children", "detail.relayout_data"]}

    # get the result ones before running the app, so we can set some default values
    # variable naming seems complex, but as these are global to the app, these complex names make sure we do not
    # accidentally reuse them
    _global_bokeh_df, _global_default_perplexity, _global_corr_thresh = create_tsne(session_id, folder_name)
    _global_measurement_types = list(_global_bokeh_df["measurement"].unique())
    _global_component_types = list(_global_bokeh_df["component"].unique())

    # get the explanation text from the file
    with open("./assets/explanation.txt") as filet:
        text = filet.read()

    # define the app layout
    layout_definition = EventListener(
        EventListener(children=[html.Div([
        dcc.Store(id="raw-signal-selected-x", data=None),  # to store the clicked data
        dcc.Store(id="signal-name-store", data=[]),  # to store signal names
        dbc.Alert(
            [
                html.H4("Too many selection. Please delete some shapes!"),
                html.Hr(),
                "Delete shapes using the [del] key.",
                html.Br(),
                "See ",
                html.A("here", href="#", className="alert-link"),
                " for details."
            ],
            id="too-many-shapes-alert",
            color="danger",
            dismissable=True,
            fade=True,
            is_open=False,
            duration=4000,
        ),
        html.Div(children=[
            html.H1(f'Change Decoupling Analyzer',
                    style={'fontSize': 40},
                    id='header'),
            "⚠ Your file does not contain anomaly scores" if anomaly_scores is None else "",
            dbc.Accordion(children=[
                dbc.AccordionItem(children=[html.Pre(text, className='pre-expl')], title="Explanation", )
            ],
                start_collapsed=True,
            ),
        ],
            style=styles['div'],
        ),
        html.Div(children=[
            dbc.Accordion(children=[
                dbc.AccordionItem(children=[
                    dcc.Graph(
                        id='histogram-graph',
                        figure=make_histogram(session_id, folder_name, _global_corr_thresh),
                    ),
                    html.Div(children=f'Correlation Value: {_global_corr_thresh}', id='corr-val-div'),
                    dcc.Slider(min=0, max=0.95,
                               value=_global_corr_thresh,
                               id='correlation-slider',
                               ),
                ],
                    title="Correlation Histogram", )
            ],
                start_collapsed=True,
            ),
        ],
            style=styles['div'],
        ),
        html.Div(children=[
            "This is the scatter plot of Signals",
            dbc.Row(children=[
                dbc.Col(children=[
                    dbc.Accordion(children=[
                        dbc.AccordionItem(children=[
                            dcc.Dropdown(
                                _global_component_types,
                                _global_component_types,
                                multi=True,
                                id='component-select',
                            )
                        ],
                            title="Measurement Selection",
                        ),
                    ],
                        start_collapsed=True,
                    ),
                ],
                ),
                dbc.Col(children=[
                    dbc.Accordion(children=[
                        dbc.AccordionItem(children=[
                            dcc.Dropdown(
                                _global_measurement_types,
                                _global_measurement_types,
                                multi=True,
                                id='measurement-select',
                            )
                        ],
                            title="Measurement Selection",
                        ),
                    ],
                        start_collapsed=True,
                    )
                ],
                ),
            ]
            ),
            html.Div(children=[
                    dbc.Row(children=[
                        dbc.Col(children=[
                            dcc.Loading(children=[
                                dcc.Graph(
                                    id='scatter-graph',
                                    figure=create_scatter(session_id, folder_name),
                                    style={'width': f'40vw', 'height': '30vw'},
                                ),
                                html.Div(id="graph-loading-signal", style={"display": "none"}),
                            ],
                                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                            )
                        ],
                        ),
                        dbc.Col(children=[
                            dcc.Loading(children=[
                                dcc.Graph(
                                    id='scatter-graph3d',
                                    figure=create_scatter_3d(session_id, folder_name),
                                    style={'width': f'40vw', 'height': '30vw'},
                                ),
                            ],
                                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                            )
                        ],
                        ),
                    ]
                    ),
            ]),
            "Perplexity value",
            dcc.Slider(1, _global_bokeh_df.shape[0] // 2, 2,
                       value=_global_default_perplexity,
                       id='perplexity-slider',
                       ),
        ],
            style= styles['div'],
            hidden=False,
        ),

        html.Div(children=[
            "Shortcuts: [DEL] deletes the last rectangle. [0] sets view to the heatmap. Any other number scrolls to "
            "the regarding selection, e.g. [1].",

                html.Div(children=[
                    dcc.Loading(
                        id="loading-signals",
                        children=[
                            dcc.Graph(
                                id=score_graph_id,
                                figure=go.Figure(go.Scatter(x=[], y=[])),
                                className='graph-class-raw-signal',
                                config = {"modeBarButtonsToAdd": ["select2d"]},
                            ),
                            html.Div(id="shape-event-dummy", style={"display": "none"}),
                            html.Div(id="shape-event-store"),
                        ],
                        type="circle",
                        overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                    ),
                ]),
            html.Button("Delete All", id="scatter-delete-button-all", className="button-4"),
            html.Button("Delete Active", id="scatter-delete-button-active", className="button-4"),
        ],
            style= styles['div'] |styles['stickydiv'],
            hidden=True,
            id='div-scatter-signal-graph',
        ),
        html.Div(children=[
            "Click on a time frame.",
            html.Div(children=[
                dcc.Loading(
                    id="loading-signals2",
                    children=[
                        dcc.Graph(
                            id='scatter-signal-graph2',
                            figure=go.Figure(go.Scatter(x=[], y=[])),
                        ),
                    ],
                    type="circle",
                    overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                ),
            ]),
        ],
            style=styles['stickydiv'] | styles['div'],
            hidden=True,
            id='div-scatter-signal-graph2',
        ),
        html.Div(children=[
        ],
            id="scatter-overall-div"
        ),
        html.Div(children=[
            dcc.Loading(
                id="loading-signals2",
                children=[
                    dcc.Graph(
                        id='raw-scatter-signal-graph',
                        figure=go.Figure(go.Scatter(x=[], y=[])),

                    ),
                ],
                type="circle",
                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
            )
        ],
            id="scatter-overall-div2",
            style=styles['div'],
            hidden=True
        ),
        html.Div(
            children=[],
            style=styles['div'],
            id='signal-include-container',
            hidden=True,
        ),
        html.Div(
            children=[],
            style=styles['div'],
            id='signal-ignore-container',
            hidden=True,
        ),
    ],
    )
    ],
        events=[delete_event], logging=True, id="scatter-delete-listener", style={"overflow": "clip"},
    ),
        events=[move_event], logging=True, id="scatter-move-listener", style={"overflow": "clip"},
    )
    return layout_definition