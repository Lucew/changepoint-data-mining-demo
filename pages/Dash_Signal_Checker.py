import logging
import time
import inspect
import itertools

import plotly.graph_objs
from dash import dcc, html, Input, Output, State, ctx, register_page, callback
import dash
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd

import util.load_data as utl
import util.styles as usty
import util.draw_heatmap as uheat
import util.draw_scatter as uscat
import util.process_kks as ukks
import pages.Dash_Heatmap_Analyzer as pageheatmap
from GLOBALS import *


# register the page to our application
register_page(__name__, path="/signal")
logger = logging.getLogger("frontend-logger")


def get_initial_figures(session_id: str, folder_name: str, target_window_size: int = None,
                        component_selection: list[str] = None, measurement_selection: list[str] = None
                        ) -> [list[tuple[str]], int, list[int], plotly.graph_objs.Figure, pd.Timestamp, pd.Timestamp, plotly.graph_objs.Figure]:
    start = time.perf_counter()

    # load the data into memory to get some information
    scores, _, window_sizes, _, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # parse the signals and only keep the ones we want to select
    signal_ids = list(scores.keys())
    signal_ids = ukks.signal_name_filter(signal_ids, component_list=component_selection, measurement_list=measurement_selection)

    # get the initial window size
    if target_window_size is None:
        target_window_size = min(window_sizes)
    elif target_window_size not in window_sizes:
        raise ValueError(f"{target_window_size=} must be in {window_sizes=}.")

    # create the tsne figure
    scatter_figure = uscat.create_scatter(session_id, folder_name, correlation_threshold=-2.0, window_size=target_window_size, selected_signals=signal_ids)
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Created completely new scatter figure in {time.perf_counter() - start:0.2f} seconds.")
    return signal_ids, target_window_size, window_sizes, scatter_figure


def layout(session_id: str = "", folder_name: str="", selection_names: dict[str:dict[str:str]] = None,
           selection_values = list[list[str]], **kwargs):

    # get the start time
    start = time.perf_counter()

    # log the request
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Requested the signal analysis page.")

    # check whether we have a folder
    if not folder_name or len(selection_values) < 3:
        return html.H1("Please upload a file using the sidebar.")

    # get the selections
    block_selection = tuple(selection_values[0])
    turbine_select = tuple(selection_values[1])
    component_selection = tuple(selection_values[2])
    measurement_selection = tuple(selection_values[3])

    # create the initial figures and infos
    signal_ids, initial_window_size, window_sizes, scatter_figure = get_initial_figures(session_id, folder_name, component_selection=component_selection, measurement_selection=measurement_selection)

    # get the signals shortly to save the minimum and maximum timestamp
    _, _, _, _, _, raw_signals_grouped, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # go through the groups and find minimum and maximum timestamps
    ts_min = raw_signals_grouped.obj.index.min()
    ts_max = raw_signals_grouped.obj.index.max()

    # get the explanation text from the file
    text = pageheatmap.get_explanation()

    # define the app layout
    layout_definition = html.Div([
        dcc.Store('signalselect-signal-name-list', data=signal_ids),
        dcc.Store('signalselect-timestamp-minmax', data=(ts_min, ts_max)),
        html.Div(children=[
            html.H1(f'Systematic Signal Analyzer',
                    style={'fontSize': 40},
                    id='signalselect-header'),
            dbc.Accordion(children=[
                dbc.AccordionItem(children=[html.Pre(text, className='pre-expl')], title="Explanation", )
            ],
                start_collapsed=True,
            ),
            "Window Size Selection",
            dcc.Loading([
                dbc.Select(id="signalselect-select-window-size", options=sorted(window_sizes), value=initial_window_size),
            ],
                type="circle",
                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                id="signalselect-windowselect-loading",
                target_components={'signalselect-scatter-graph': 'figure'}
            ),
        ],
            style=usty.div_styles['div'],
        ),
        dcc.Loading(children=[
            html.Div(children=[
                dbc.Row(children=[
                    dbc.Col(children=[
                        html.H3("Signal Scatter Plot"),
                        dcc.Loading(children=[
                            dbc.Accordion(children=[
                                dbc.AccordionItem(children=[
                                    dcc.Graph(
                                        id='signalselect-scatter-graph',
                                        figure=scatter_figure,
                                        style={'width': f'40vw', 'height': '25vw', 'marginLeft': 'auto', 'marginRight': 'auto'},
                                    ),
                                ],
                                    title="Scatter Plot",
                                ),
                            ],
                                start_collapsed=False,
                            ),
                        ],
                            type="circle",
                            overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                            id='signalselect-data-loader',
                        ),
                    ]),
                ]),
            ],
                style=usty.div_styles['div'],
                id='signalselect-column-container'
            ),
            dcc.Loading([
                html.Div(children=[
                ],
                    id="signalselect-raw-signal-graph-container",
                ),
            ],
                type="circle",
                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
            ),
        ],
            type="circle",
            overlay_style={"visibility": "visible", "filter": "blur(2px)"},
            id="signalselect-all-graph-loading",
            target_components={'signalselect-scatter-graph': 'figure'}
        ),
    ],
    )

    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Build the signal analysis page in {time.perf_counter() - start:0.2f} s.")
    return layout_definition


# app callbacks --------------------------------------------------------------------------------------------------------


@callback(
    Output('signalselect-scatter-graph', 'figure'),
    State("session-id", "data"),
    State("folder-name", "data"),
    Input("signalselect-select-window-size", "value"),
    running=[
        (Output("signalselect-data-loader", "display"), "show", "auto"),
        (Output("signalselect-windowselect-loading", "display"), "show", "auto"),
    ],
    prevent_initial_call=True,
)
def redraw_scatter_graph(session_id: str, folder_name: str, window_size: str):


    # write to logger
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] We adapt the scatter plot ({window_size=}).")

    # do not redraw the scatter if we just reordered
    new_figure = dash.no_update
    if ctx.triggered_id == "signalselect-select-window-size":
        # create the new figure
        new_figure = uscat.create_scatter(session_id, folder_name, correlation_threshold=-2.0,
                                          window_size=int(window_size))

    # redraw the scatter
    return new_figure


@callback(
    Output(component_id="signalselect-raw-signal-graph-container", component_property='children'),
    State("session-id", "data"),
    State("folder-name", "data"),
    State("signalselect-signal-name-list", "data"),
    State("signalselect-timestamp-minmax", "data"),
    State("signalselect-select-window-size", "value"),
    Input('signalselect-scatter-graph', 'clickData'),
    Input('signalselect-scatter-graph', 'selectedData'),
    running=[
        (Output("signalselect-data-loader", "display"), "show", "auto"),
        (Output("signalselect-windowselect-loading", "display"), "show", "auto"),
    ],
    prevent_initial_call=True,
)
def create_signal_graph(session_id: str, folder_name: str, signals_ids: list[str], timestamps: tuple[str, str], window_size: int, clicked_data, selected_data):

    # transform the window size
    window_size = int(window_size)

    # get the keys of the ids
    prop_id = list(ctx.triggered_prop_ids.keys())[0]

    # get the points if we clicked or select
    if prop_id.endswith('selectedData') and selected_data:
        point_data = selected_data
    elif prop_id.endswith('clickData') and clicked_data:
        point_data = clicked_data
    else:
        raise PreventUpdate

    # extract the signal names
    signal_names = [point["customdata"][0] for point in point_data["points"]]
    if not signal_names:
        return []

    # create the raw signal figure
    # check whether we want to draw too many signals per shape
    if len(signal_names) > 3*RAW_SIGNAL_PLOT_MAXIMUM_NUMBER:
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Too many signals: ({len(signal_names)=}).")
        fig = uheat.create_empty_figure_with_text(f"Too many signals ({len(signal_names)=} > {RAW_SIGNAL_PLOT_MAXIMUM_NUMBER}) in this selection.")
    # create the figure from the newly selected data
    else:
        shape = {'x0': timestamps[0], 'x1': timestamps[1]}
        fig, _, _, _ = uheat.create_raw_signal_figure(session_id, folder_name, shape, 0, len(signal_names), signal_names, window_size)
        fig.update_layout(hovermode="x unified")
    raw_signal_graph = dash.dcc.Graph(figure=fig, id="signalselection-raw-signal-graph")

    # create the new div
    new_raw_plot = dash.html.Div(children=[
        dcc.Loading([
        dash.html.H3("Selected Signals"),
        dash.html.Details(children=[
            dash.dcc.Loading(children=[
                dash.html.Div(children=[
                    raw_signal_graph,
                ],
                    id="signalselection-div-raw-signal-graph",
                )
            ],
                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
            )
        ],
            open=True,
            id="signalselection-raw-signal-div",
        )
        ],
            type="circle",
            overlay_style={"visibility": "visible", "filter": "blur(2px)"},
        ),
        ],
        style=usty.div_styles['div'],
        id=f'signalselection-scatter-signal-selection-div',
    )
    return new_raw_plot