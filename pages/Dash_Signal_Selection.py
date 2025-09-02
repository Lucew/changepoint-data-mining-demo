import logging
import time
from time import perf_counter

import dash
from dash import dcc, html, Input, Output, callback, State, clientside_callback, ctx, register_page, ClientsideFunction
from dash_extensions import EventListener
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pandas.api.typing as pdtypes
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


# define some styles for different html elements
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'color': 'black'
    },
    'div': {
        'padding': '.6rem',
        'width': '9'
                 '0%',
        'margin': 'auto',
        'boxShadow': 'dimgrey 4px 4px 2px',
        'border-radius': '10px',
        'backgroundColor': 'white',
        'marginTop': '1rem',
    },
    'stickydiv': {
        'position': 'sticky',
        'top': '-10px',
        "z-index": "20",
    },
    'dropdown': {
        'margin': 'auto',
        'width': '50%',
        'border-radius': '10px',
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
            'eraseshape'
        ]
    )
    fig.update_xaxes(scaleanchor=False)

    # Add modebar buttons
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
    regression_results, _, complete_max_correlation = utl.preprocess_regression_results(session_id, folder_name)

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
    filtered_regression_results.loc[:, "distance"] = 10_000 - filtered_regression_results["correlation"]*1000
    filtered_distance_matrix = filtered_regression_results.pivot(index='x', columns='y', values='distance').fillna(0)

    # return the filtered regression results
    logger.info(f"[{__name__}] Filtered regression results in {time.perf_counter() - start:0.2f} s.")
    return filtered_regression_results, filtered_distance_matrix, correlation_threshold


@ucache.lru_cache(maxsize=2)
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
    _, _, max_correlation = utl.preprocess_regression_results(session_id, folder_name)

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
                             'Anomaly Score': anomaly_scores.loc[distance_matrix.columns, 'score'],
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
    rec_anomaly = bokeh_df.loc[:, 'Anomaly Score'].to_numpy()
    rec_anomaly[np.isnan(rec_anomaly)] = 0
    zeroless_mask = rec_anomaly != 0
    zero_mask = ~zeroless_mask
    rec_zeroless = np.log(rec_anomaly[zeroless_mask])
    rec_anomaly[zero_mask] = rec_zeroless.min()
    rec_anomaly[zeroless_mask] = rec_zeroless
    rec_anomaly = (rec_anomaly - rec_anomaly.min()) / (rec_anomaly.max() - rec_anomaly.min())
    bokeh_df['opac'] = rec_anomaly

    logger.info(f"[{__name__}] Created TSNE ({perf_counter()-started:0.2f} s).")
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
    """
    fig.update_layout(
        dragmode='lasso',  # define dragmode
    )
    """
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
    _, _, complete_max_correlation = utl.preprocess_regression_results(session_id, folder_name)

    # make the histogram out of it
    fig = px.ecdf(complete_max_correlation, x="correlation", marginal="histogram", ecdfnorm=None)
    mask = complete_max_correlation["correlation"] > correlation_threshold
    fig.add_vline(correlation_threshold, annotation_text=f'Remaining Tags: {mask.sum()}')
    return fig


# app callbacks --------------------------------------------------------------------------------------------------------

@callback(
    Output(component_id="scatter-signal-graph", component_property="figure", allow_duplicate=True),
    Output(component_id='div-scatter-signal-graph', component_property="hidden"),
    Output(component_id='scatter-overall-div', component_property='hidden'),
    Input("session-id", "data"),
    Input("folder-name", "data"),
    Input(component_id="scatter-graph", component_property="selectedData"),
    prevent_initial_call=True)
def select_signals_scatter(session_id: str, folder_name: str, selected_data):

    # check whether we selected nothing
    if selected_data is None or len(selected_data["points"]) == 0:
        return go.Figure(go.Scatter(x=[], y=[])), True, True

    # get the signals we selected
    selected_signals = [point["customdata"][0] for point in selected_data["points"]]

    # get the regression results
    regression_results, _, _ = utl.preprocess_regression_results(session_id, folder_name)

    # get the scores from the files
    scores, _, _, _, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # compute the weighted scoring for each of the selected signals
    # TODO mean correlation value per cluster as hint whether it is a good cluster
    result_df = procd.compute_weighted_residual_norm(regression_results, selected_signals, scores, coming_from='signal-selection')

    # make the figure from the signals
    fig = draw_heatmap(result_df)

    return fig, False, False


@callback(
    Output(component_id="scatter-signal-graph2", component_property="figure", allow_duplicate=True),
    Output(component_id="raw-scatter-signal-graph", component_property="figure", allow_duplicate=True),
    Output(component_id='scatter-overall-div2', component_property='hidden'),
    Output(component_id='div-scatter-signal-graph2', component_property='hidden'),
    Input("session-id", "data"),
    Input("folder-name", "data"),
    Input(component_id="scatter-graph", component_property="clickData"),
    prevent_initial_call=True)
def click_signals_scatter(session_id: str, folder_name: str, click_data):

    # check whether we selected nothing
    if click_data is None:
        return go.Figure(go.Scatter(x=[], y=[])), go.Figure(go.Scatter(x=[], y=[])), True, True

    # get the signals we selected
    selected_signal = [point["customdata"][0] for point in click_data["points"]][0]

    # get the regression results
    _, extended_regression_results_grouped, _ = utl.preprocess_regression_results(session_id, folder_name)

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

    return fig, fig2, False, False


@callback(
    Output(component_id="scatter-signal-graph2", component_property="figure", allow_duplicate=True),
    Output(component_id="raw-scatter-signal-graph", component_property="figure", allow_duplicate=True),
    Input(component_id="scatter-signal-graph2", component_property="clickData"),
    State(component_id="scatter-signal-graph2", component_property="figure"),
    State(component_id="raw-scatter-signal-graph", component_property="figure"),
    prevent_initial_call=True)
def click_in_residuals(click_data, residual_fig, signal_fig):

    # check whether we selected nothing
    if click_data is None:
        return dash.no_update, dash.no_update

    # delete the old shapes
    residual_fig['layout']['shapes'] = []
    signal_fig['layout']['shapes'] = []

    # get the signals we selected
    selected_time = [point["x"] for point in click_data["points"]][0]

    # create add a shape where we clicked
    residual_fig = go.Figure(residual_fig)
    signal_fig = go.Figure(signal_fig)
    residual_fig.add_vline(x=selected_time)
    signal_fig.add_vline(x=selected_time)

    return residual_fig, signal_fig


@callback(
    Output('scatter-graph', 'figure'),
    Output('scatter-graph3d', 'figure'),
    Input("session-id", "data"),
    Input("folder-name", "data"),
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
    Input("session-id", "data"),
    Input("folder-name", "data"),
    Input('correlation-slider', 'value')
)
def update_histogram(session_id: str, folder_name: str, correlation_threshold: float):
    return make_histogram(session_id, folder_name, correlation_threshold), f'Correlation Value: {correlation_threshold}'


@callback(
    Output(component_id='scatter-signal-graph', component_property='figure', allow_duplicate=True),
    Output(component_id='scatter-overall-div', component_property='children'),
    Output(component_id='scatter-delete-button-all', component_property='children'),
    Output(component_id="scatter-delete-listener", component_property="event"),
    Input("session-id", "data"),
    Input("folder-name", "data"),
    Input(component_id='scatter-delete-button-all', component_property='n_clicks'),
    State(component_id='scatter-signal-graph', component_property='figure'),
    Input(component_id='scatter-signal-graph', component_property='relayoutData'),
    Input(component_id='scatter-overall-div', component_property='children'),
    Input(component_id="scatter-delete-listener", component_property="n_events"),
    State(component_id="scatter-delete-listener", component_property="event"),
    Input(component_id='scatter-signal-graph', component_property='clickData'),
    prevent_initial_call=True
)
def delete_shapes(session_id: str, folder_name: str,
                  delete_all, current_figure, relayoutData, all_children, n_delete_events, delete_event_data,
                  click_event):

    # get existing shapes and sort out all that are not rect
    if len(current_figure['data'][0]['y']):
        shapes = [ele for ele in current_figure['layout'].get('shapes', []) if ele['type'] == 'rect']
    else:
        shapes = []

    # check for active shapes
    if delete_event_data:

        # get the index of the active shape
        active_idx = delete_event_data['detail.children']
        assert len(active_idx) <= 1, f'There are more than one active shape {active_idx}.'

        # get the active shape or delete the last shape
        if active_idx:
            active_idx = int(active_idx[0])
        else:
            active_idx = len(shapes)-1

        # build the new shapes
        shapes = [ele for idx, ele in enumerate(shapes) if idx != active_idx]

    # check whether we pressed the delete button
    if 'scatter-delete-button-all' == ctx.triggered_id:
        shapes.clear()

    # update the shape accordingly (by snapping to complete integers) and only keep shapes that we like
    for idx, shape in enumerate(shapes):

        # set the indices
        shape['y0'] = round(shape['y0'])
        shape['y1'] = round(shape['y1'])

        # give the shape a text
        shape['label']['text'] = f'Selection {idx + 1}'
        shape['label']['font'] = {'color': 'white'}

    # check whether we have shapes
    if shapes:
        # load the raw signals
        _, _, _, _, _, signals, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))
    else:
        signals = None

    # get the y-axis labels
    column_names = current_figure["data"][0]["y"]

    # check the click event
    click_point = None
    if click_event is not None:
        click_point = pd.Timestamp(click_event['points'][0]['x'])

    # make the other plots for each of the shapes
    all_children.clear()
    for idx, shape in enumerate(shapes):

        # get all the columns we need to select (depending on where we start, one of the shapes is the larger one)
        column_start = shape['y0']
        column_end = shape['y1']
        column_start, column_end = sorted((column_start, column_end))

        # get the time we need to select (depending on where we start, one of the shapes is the larger one)
        time_start = pd.Timestamp(shape['x0'])
        time_end = pd.Timestamp(shape['x1'])

        # make a figure using the first signal
        names = set(column_names[column_start:column_end+1])
        tmp_signals = pd.concat(signals.get_group(grp).loc[time_start:time_end] for grp in names)
        # tmp_signals = signals[signals['sensor'].isin(names)].loc[time_start:time_end]
        fig = px.line(tmp_signals, y='normalized value', line_group='sensor', color='sensor', custom_data='value', hover_data=['sensor', 'value'], markers=True)

        # mark a line if the click point is in the corresponding axis
        selection_text = None
        if click_event is not None and time_start <= click_point <= time_end:
            fig.add_vline(x=click_point)
            selection_text = f'You selected: {click_event["points"][0]["y"]}   @{click_point}'

        all_children.append(html.Div(children=[html.H3(f"Signals for Selection {idx+1}"),
                                               selection_text,
                                               html.Details(children=[
                                                                html.Div(children=[
                                                                    dcc.Graph(figure=fig)])
                                                            ],
                                                            open=True,
                                                            id=f'scatter-signal-selection{idx + 1}'
                                                            )]
                                     , style=styles['div']
                                     )
                            )

    # update figure layout (shapes and also the ordering on the yaxis)
    current_figure['layout']['shapes'] = shapes
    current_figure = go.Figure(current_figure)

    return current_figure, all_children, f"Delete all {len(shapes)}", None


# how to call an update from button presses modified from
# https://community.plotly.com/t/how-to-get-trigger-from-keyboard/76960/6
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="delete_stuff_scatter"),
    Output("scatter-signal-graph", "id"),
    Input("scatter-signal-graph", "id")
)


# dash app layout ------------------------------------------------------------------------------------------------------
def layout(session_id: str, folder_name: str, **kwargs):

    # check whether we have a folder
    if not folder_name:
        return html.H1("Please upload a file using the sidebar.")

    # check whether we have all necessary data
    _, _, _, anomaly_scores, distances, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))
    if anomaly_scores is None or distances is None:
        return html.H1("Your zip-file does not contain the anomaly_scores.parquet or/and the distances.csv file(s).")

    # define an event for our custom event listener to drag and drop buttons
    drop_event = {"event": "dropcomplete", "props": ["detail.name", "detail.children"]}
    delete_event = {"event": 'shapeDeletion', "props": ["detail.children"]}

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
    layout_definition = html.Div([
        html.Div(children=[
            html.H1(f'Change Decoupling Analyzer',
                    style={'fontSize': 40},
                    id='header'),
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
                               persistence="session"),
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
                                persistence="session",
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
                                persistence="session"
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
                dcc.Loading(
                    id="loading-scatter",
                    children=[
                        dbc.Row(children=[
                            dbc.Col(children=[
                                dcc.Graph(
                                    id='scatter-graph',
                                    figure=create_scatter(session_id, folder_name),
                                    style={'width': f'40vw', 'height': '30vw'},
                                ),
                            ],
                            ),
                            dbc.Col(children=[
                                dcc.Graph(
                                    id='scatter-graph3d',
                                    figure=create_scatter_3d(session_id, folder_name),
                                    style={'width': f'40vw', 'height': '30vw'},
                                ),
                            ],
                            ),
                        ]
                        ),
                    ],
                    type="circle",
                ),
            ]),
            "Perplexity value",
            dcc.Slider(1, _global_bokeh_df.shape[0] // 2, 2,
                       value=_global_default_perplexity,
                       id='perplexity-slider',
                       persistence="session"),
        ],
            style=styles['div'],
            hidden=False,
        ),
        html.Div(children=[
            "Shortcuts: [DEL] deletes the last rectangle. [0] sets view to the heatmap. Any other number scrolls to "
            "the regarding selection, e.g. [1].",
            EventListener(children=[
                html.Div(children=[
                    dcc.Loading(
                        id="loading-signals",
                        children=[
                            dcc.Graph(
                                id='scatter-signal-graph',
                                figure=go.Figure(go.Scatter(x=[], y=[])),
                            ),
                        ],
                        type="circle",
                    ),
                ]),
            ],
                events=[delete_event], logging=True, id="scatter-delete-listener",
            ),
            html.Button("Delete All", id="scatter-delete-button-all", className="button-4"),
        ],
            style=styles['stickydiv'] | styles['div'],
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
            dcc.Graph(
                id='raw-scatter-signal-graph',
                figure=go.Figure(go.Scatter(x=[], y=[])),
            ),
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
    return layout_definition