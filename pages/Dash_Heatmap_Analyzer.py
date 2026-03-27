import logging
import time
import inspect

import plotly.graph_objs
from dash import dcc, html, Input, Output, State, ctx, ClientsideFunction, register_page, clientside_callback, callback, stringify_id, Patch, ALL
import dash
from dash.exceptions import PreventUpdate
from dash_extensions import EventListener
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

import util.load_data as utl
import util.prepocessing as prep
import util.cache_registry as ucache
import util.styles as usty
import util.draw_heatmap as uheat
import util.draw_scatter as uscat
from GLOBALS import *


# register the page to our application
register_page(__name__, path="/heatmap")
logger = logging.getLogger("frontend-logger")

# define a type for the graphs that should show a line on click
on_click_type = "heatmap-raw-signal-graph"

# set an id the heatmap graph
heatmap_index = 'heatmap-signal-graph'
heatmap_id = {"type": on_click_type, "index": heatmap_index}



@ucache.lru_cache(maxsize=CACHE_SIZE)
def process_signals(session_id: str, folder_name: str, window_size: str = None, signal_list: tuple[str] = None,
                    normalization_window_size: int = None) -> tuple[pd.DataFrame, int, tuple[int]]:
    start = time.perf_counter()
    assert signal_list is None or len(signal_list) > 0, 'You have to select signals.'

    # get the data from memory
    scores, _, window_sizes, _, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # get the smallest window size if None is given
    if window_size is None:
        window_size = min(window_sizes)
    window_size = int(window_size)

    # select the signals
    if signal_list:
        signal_list = set(signal_list)
        scores = {name: df for name, df in scores.items() if name in signal_list}

    # get the score for the defined window size
    scores = {name: df.get_group(window_size)[["value"]].rename(columns={'value': name}) for name, df in scores.items()}
    score_df = pd.concat(scores.values(), axis=1)

    # normalize the score
    score_df = prep.normalization(score_df, window_length=normalization_window_size)

    # make the reference signal subtraction
    score_df = score_df.sub(score_df.mean(axis=1), axis="index").abs()
    # TODO: do not absolute value the negative scores

    # transform the index into a pandas timestamp
    score_df.index = pd.to_datetime(score_df.index)

    logger.info(f"[{__name__}] Preprocessed data in {time.perf_counter() - start:0.2f} s. Parameters: {window_size=}, {signal_list=}, {normalization_window_size=}")
    return score_df, window_size, window_sizes


def create_heatmap(session_id: str, folder_name: str, window_size: int = None, signal_list: tuple[str,...] = None,
                   normalization_window_size: int = None) -> [plotly.graph_objs.Figure, pd.Timestamp, pd.Timestamp]:
    # read the signals
    score_df, _, _ = process_signals(session_id=session_id, folder_name=folder_name, window_size=window_size, signal_list=signal_list, normalization_window_size=normalization_window_size)

    # make the figure
    fig = uheat.draw_heatmap(score_df)
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(score_df.columns),
        constrain='range',
    )
    return fig, score_df.index.min(), score_df.index.max()


@ucache.lru_cache(1)
def get_explanation():
    # get the explanation text from the file
    with open("./assets/explanation.txt") as filet:
        text = filet.read()
    return text

def make_signal_selection_store(event_type: [None | str], selected_signals: list[str]) -> dict:
    return {'event-type': event_type, 'selected-signals': selected_signals}

def unpack_signal_selection_store(signal_selection_store: dict) -> [str, list[str]]:
    """
    :param signal_selection_store:
    :return: the recent event type, list of selected signals
    """
    return signal_selection_store['event-type'], signal_selection_store['selected-signals']

def make_shape_store_entry(xmin: str, xmax: str, initial_shapes: [None | list[dict]] = None):
    initial_shapes = [] if initial_shapes is None else initial_shapes
    return {'range': (xmin, xmax), 'shapes': initial_shapes}

def unpack_shape_store_entry(shape_store_entry: dict) -> [[str, str], list[dict]]:
    return shape_store_entry['range'], shape_store_entry['shapes']

# dash app layout ------------------------------------------------------------------------------------------------------

# delete all shapes
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="delete_all_shapes"),
    Output("heatmap-delete-button-all", "id"),
    Input("heatmap-delete-button-all", "id"),
    Input('heatmap-container', "id")
)

# delete active shapes
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="delete_active_shape"),
    Output("heatmap-delete-button-active", "id"),
    Input("heatmap-delete-button-active", "id"),
    Input('heatmap-container', "id")
)

# how to call an update from button presses modified from
# https://community.plotly.com/t/how-to-get-trigger-from-keyboard/76960/6
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="button_press_interaction"),
    Output(heatmap_id, "id"),
    Input('heatmap-container', "id"),
    State("heatmap-raw-signal-graph-container", "id"),
)

# how to make draggable signals
# https://community.plotly.com/t/ordering-children-with-drag-drop-how-to-trigger-dash-component-has-changed-from-js/47243/10
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="make_draggable"),
    Output("heatmap-drag-container", "data-drag"),
    [Input("heatmap-drag-container", "id"),
     Input("heatmap-drag-ignore-container", "id")],
)

# textbox selection with regex
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="select_signals"),
    Output("heatmap-signal-include-container", "children"),
    [Input("heatmap-signal-include-text-search", "value"),
     Input("heatmap-drag-container", "id"),
     Input("heatmap-drag-ignore-container", "id"),
     Input("heatmap-drag-container", "id")
     ],
)
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="select_signals"),
    Output("heatmap-signal-ignore-container", "children"),
    [Input("heatmap-signal-ignore-text-search", "value"),
     Input("heatmap-drag-ignore-container", "id"),
     Input("heatmap-drag-container", "id"),
     Input("heatmap-drag-container", "id")
     ],
)


def make_selection_accordion(signal_ids: list[str]):
    # define an event for our custom to drag and drop buttons
    drop_event = {"event": "dropcomplete", "props": ["detail.name", "detail.children"]}

    signal_selection_accordion = dbc.Accordion(children=[
        dbc.AccordionItem(children=[
            EventListener(children=[
                dbc.Row(children=[
                    dbc.Col(children=[
                        'Ignore signals with pattern: ',
                        dcc.Input(
                            id="heatmap-signal-include-text-search",
                            type='text',
                            placeholder="Signal Regex",
                            debounce=True,
                        ),
                    ]),
                    dbc.Col(children=[
                        'Plot signals with pattern: ',
                        dcc.Input(
                            id="heatmap-signal-ignore-text-search",
                            type='text',
                            placeholder="Signal Regex",
                            debounce=True,
                        ),
                    ]),
                ]),
                html.Br(),
                dbc.Row(children=[
                    dbc.Col(children=[
                        dbc.Card(children=[
                            dbc.CardHeader(["Plotted Columns"]),
                            dbc.CardBody(children=
                            [
                                html.Div(col, id=f"heatmap-signal-{col}", disable_n_clicks=True,className='a')
                                for col in signal_ids
                            ],
                                id='heatmap-drag-container', style={"minHeight": "10px"}),
                        ]),
                    ]),
                    dbc.Col(children=[
                        dbc.Card(children=[
                            dbc.CardHeader(["Ignored Columns"]),
                            dbc.CardBody(children=[
                                html.Div(children=[], id='heatmap-drag-ignore-container', style={"minHeight": "10px"})
                            ]),
                        ]),
                    ]),
                ]),
            ],
                events=[drop_event], logging=False, id="heatmap-drag-event-listener",
            ),
            html.Br(),
            dcc.Input(
                id="heatmap-normalization-input",
                type="number",
                placeholder="Z-Score window size",
                debounce=True,
            ),
        ],
            title="Sort and Select columns by dragging and dropping their column names.",
        ),
    ],
        start_collapsed=True,
    )
    return signal_selection_accordion

def layout(session_id: str = "", folder_name: str="", **kwargs):

    # check whether we have a folder
    if not folder_name:
        return html.H1("Please upload a file using the sidebar.")

    # load the data into memory to get some information
    scores, _, window_sizes, _, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))
    score_df, _, _ = process_signals(session_id=session_id, folder_name=folder_name, window_size=min(window_sizes))
    window_size = min(window_sizes)

    # check whether there are too many sensors
    if len(scores) > 100:
        return html.H1(f"There are too many sensors ({len(scores)=}) reduce to 50 or less.")

    # create the heatmap figure
    figure, start_time, end_time = create_heatmap(session_id=session_id, folder_name=folder_name, window_size=window_size)
    logger.info(f"[{__name__}] Created initial figure and window size options.")

    # define a custom event for the shape deletion
    delete_event = {"event": 'shapeDeletion', "props": ["detail.children"]}

    # get the explanation text from the file
    text = get_explanation()

    # define the initially used signals
    initial_signal_ids = list(score_df.columns)

    # define the app layout
    layout_definition = html.Div([
        dcc.Store(id='heatmap-all-signal-store', data=initial_signal_ids.copy()),
        dcc.Store(id='heatmap-active-signal-store', data=make_signal_selection_store(None, initial_signal_ids)),
        dcc.Store(id='heatmap-displayed-signal-store', data=initial_signal_ids),
        dcc.Store(id="heatmap-figure-shape-store", data={stringify_id(heatmap_id): make_shape_store_entry(start_time, end_time)}),
        html.Div(children=[
            html.H1(f'Change Decoupling Heatmap Analyzer',
                    style={'fontSize': 40},
                    id='heatmap-header'),
            dbc.Accordion(children=[
                dbc.AccordionItem(children=[html.Pre(text, className='pre-expl')], title="Explanation", )
            ],
                start_collapsed=True,
            ),
            "Window Size Selection",
            dbc.Select(id="heatmap-select-window-size", options=sorted(window_sizes), value=window_size),
        ],
            style=usty.div_styles['div'],
        ),
        html.Div(children=[
            dbc.Row(children=[
                dbc.Col(children=[
                    html.H3("Signal Scatter Plot"),
                    dcc.Loading(children=[
                        dbc.Accordion(children=[
                            dbc.AccordionItem(children=[
                                dcc.Graph(
                                    id='heatmap-scatter-graph',
                                    figure=uscat.create_scatter(session_id, folder_name, correlation_threshold=-2.0, window_size=window_size, selected_signals=initial_signal_ids),
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
                        id={'type': 'heatmap-data-loader', 'index': 'signal-selection'},
                    ),
                ]),
                dbc.Col(children=[
                    html.H3("Signal Selection"),
                    dcc.Loading(children=[
                        make_selection_accordion(initial_signal_ids)
                    ],
                        type="circle",
                        overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                        id={'type': 'heatmap-data-loader', 'index': 'signal-selection'},
                    ),
                ]),
            ]),
        ],
            style=usty.div_styles['div'],
            id='heatmap-column-container'
        ),
        EventListener(children=[
            html.Div(children=[
                "Shortcuts: [DEL] deletes the last drawn rectangle. [0] sets view to anomaly_heatmap. Any other number scrolls to "
                "the regarding selection, e.g. [1].",
                html.Div(children=[
                    dcc.Loading(children=[
                        dcc.Graph(
                            id=heatmap_id,
                            figure=figure,
                        ),
                    ],
                        type="circle",
                        overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                        id={'type': 'heatmap-data-loader', 'index': 'heatmap-graph'},
                    ),
                ]),
                html.Button("Delete All", id="heatmap-delete-button-all", className="button-4"),
                html.Button("Delete Active", id="heatmap-delete-button-active", className="button-4"),
            ],
                style=usty.div_styles['div'] | usty.div_styles['stickydiv'],
                id='heatmap-container',
            ),
        ],
            events=[delete_event], logging=True, id="heatmap-delete-listener",
        ),
        html.Div(children=[
        ],
            id="heatmap-raw-signal-graph-container",
        ),
        html.Div(
            children=[],
            style=usty.div_styles['div'],
            id='heatmap-text-container',
            hidden=True,
        ),
        html.Div(
            children=[],
            style=usty.div_styles['div'],
            id='heatmap-signal-include-container',
            hidden=True,
        ),
        html.Div(
            children=[],
            style=usty.div_styles['div'],
            id='heatmap-signal-ignore-container',
            hidden=True,
        ),
    ],
    )
    return layout_definition


# app callbacks --------------------------------------------------------------------------------------------------------

@callback(
    Output('heatmap-active-signal-store', 'data'),
    State("heatmap-drag-event-listener", "event"),
    Input("heatmap-drag-event-listener", "n_events"),
    prevent_initial_call=True,
)
def reorder_columns(event_data: dict, n_events: int):
    # get the selected signals from the event data
    selected_signals = [ele.replace('heatmap-signal-', '') for ele in event_data['detail.children']]
    return make_signal_selection_store(event_data['detail.name'], selected_signals)


@callback(
    Output('heatmap-scatter-graph', 'figure'),
    State("session-id", "data"),
    State("folder-name", "data"),
    Input("heatmap-select-window-size", "value"),
    Input('heatmap-active-signal-store', 'data'),
    prevent_initial_call=True,
)
def redraw_scatter_graph(session_id: str, folder_name: str, window_size: str, signal_store: dict):

    # unpack the signal store data
    event_type, selected_signals = unpack_signal_selection_store(signal_store)

    # write to logger
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] We adapt the scatter plot ({event_type=}).")

    # do not redraw the scatter if we just reordered
    new_figure = dash.no_update
    if event_type == 'deleteEvent':
        # create the new figure
        new_figure = uscat.create_scatter(session_id, folder_name, correlation_threshold=-2.0,
                                          window_size=int(window_size),
                                          selected_signals=selected_signals)

    # redraw the scatter
    return new_figure


@callback(
    Output(heatmap_id, 'figure', allow_duplicate=True),
    Output('heatmap-displayed-signal-store', 'data'),
    State("session-id", "data"),
    State("folder-name", "data"),
    State('heatmap-all-signal-store', "data"),
    Input("heatmap-select-window-size", "value"),
    Input("heatmap-normalization-input", "value"),
    Input('heatmap-active-signal-store', 'data'),
    Input('heatmap-scatter-graph', 'selectedData'),
    prevent_initial_call=True,
)
def modify_heatmap_content(session_id: str, folder_name: str, all_signal_names: list[str,...], window_size: str, normalization_window_size: str, signal_store: dict, scatter_select: dict):

    # make a patch object
    heatmap_patch = Patch()

    # make an output for the displayed signal store
    displayed_signals = signal_store

    # write to logger
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] We have to work on the heatmap ({ctx.triggered_id=}).")

    # check which element triggered the function
    if ctx.triggered_id == "heatmap-select-window-size" or ctx.triggered_id == "heatmap-normalization-input":

        # make a new heatmap
        new_heatmap, _, _ = create_heatmap(session_id, folder_name, window_size=int(window_size), normalization_window_size=int(normalization_window_size) if normalization_window_size else None)

        # extract the image data and put it into the patch
        heatmap_patch["data"][0]["z"] = new_heatmap["data"][0]["z"]

    elif ctx.triggered_id == 'heatmap-active-signal-store' or ctx.triggered_id == 'heatmap-scatter-graph':

        # unpack the signal store
        event_type, selected_signals = unpack_signal_selection_store(signal_store)

        # if we made a selection from the scatter graph
        if scatter_select is not None and 'range' in scatter_select:
            # get the signals we selected from the scatterplot
            sub_selected_signals = set([point["customdata"][0] for point in scatter_select["points"]])
            selected_signals = [sig for sig in selected_signals if sig in sub_selected_signals]

        # get the visible and hidden data
        visible = selected_signals
        visible_set = set(selected_signals)
        hidden = [sig for sig in all_signal_names if sig not in visible_set]
        ordered = visible + hidden
        heatmap_patch["layout"]["yaxis"]["categoryarray"] = ordered

        # normal categorical axis
        heatmap_patch["layout"]["yaxis"]["autorange"] = False
        range_end = len(visible) if visible else -0.5
        heatmap_patch["layout"]["yaxis"]["range"] = [range_end - 0.5, -0.5]
        # heatmap_patch["layout"]["yaxis"]["range"][0] = len(visible) - 0.5

        # update the displayed signals
        displayed_signals = visible
    else:
        # write to logger
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Unknown trigger id: {ctx.triggered_id=}.")

    # in all cases, we have to delete the shapes
    del heatmap_patch['layout']['shapes'][0]
    return heatmap_patch, displayed_signals


@callback(
    Output(component_id=heatmap_id, component_property='figure', allow_duplicate=True),
    Output(component_id="heatmap-delete-listener", component_property="event", allow_duplicate=True),
    Output(component_id="heatmap-raw-signal-graph-container", component_property="children", allow_duplicate=True),
    Output("heatmap-figure-shape-store", "data", allow_duplicate=True),
    Input(component_id="heatmap-delete-listener", component_property="n_events"),
    State(component_id="heatmap-delete-listener", component_property="event"),
    State(component_id={"type": on_click_type, "index": ALL}, component_property='id'),
    State("heatmap-figure-shape-store", "data"),
    State(component_id=heatmap_id, component_property='id'),
    prevent_initial_call=True
)
def delete_shapes(n_delete_events, delete_event_data, raw_signal_fig_ids: list, figure_shapes: dict[str: list], heatmap_id_dict: dict):
    figure_shape_patch, scatter_overall_patch, figure_shapes = uheat.delete_shapes(delete_event_data, raw_signal_fig_ids, figure_shapes, heatmap_id_dict)
    return figure_shape_patch, None, scatter_overall_patch, figure_shapes


@callback(
    Output(component_id=heatmap_id, component_property='figure', allow_duplicate=True),
    Output(component_id={"type": "div-raw-signal-graph", "index": ALL}, component_property='children', allow_duplicate=True),
    Output(component_id="heatmap-raw-signal-graph-container", component_property='children', allow_duplicate=True),
    Output(component_id="heatmap-figure-shape-store", component_property="data", allow_duplicate=True),
    State(component_id="session-id", component_property="data"),
    State(component_id="folder-name", component_property="data"),
    State(component_id='heatmap-displayed-signal-store', component_property='data'),
    State(component_id="heatmap-figure-shape-store", component_property="data"),
    State(component_id={"type": on_click_type, "index": ALL}, component_property='id'),
    Input(heatmap_id, "relayoutData"),
    running=[(Output({'type': 'heatmap-data-loader', 'index': 'heatmap-graph'}, "display"), "show", "auto")], # this deactivates the figure while running our function
    prevent_initial_call=True
)
def modify_shapes(session_id: str, folder_name: str, signal_names: dict[str:list[str]], figure_shapes: dict[str:list], raw_signal_figure_ids: list[str], relayout_data: dict):

    # check whether we want to draw a new shape
    if 'autosize' in relayout_data:
        raise PreventUpdate

    is_new_shape = 'shapes' in relayout_data
    is_shape_redraw = any(key.startswith('shapes[') for key in relayout_data.keys())
    if not is_new_shape and not is_shape_redraw:
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {ctx.triggered_id}. Unknown event type: {relayout_data=}.")
        raise PreventUpdate
    print(raw_signal_figure_ids)
    # make a logging entry
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {ctx.triggered_id}, {is_new_shape=}, {is_shape_redraw=}.")

    # make the processing depending on the event
    if is_new_shape:
        figure_shape_patch, raw_signal_plot_patch_list, raw_signal_plot_collection_patch, figure_shapes = uheat.create_new_raw_signal_plot(session_id, folder_name, signal_names, figure_shapes, raw_signal_figure_ids, relayout_data, on_click_type)
    elif is_shape_redraw:
        figure_shape_patch, raw_signal_plot_patch_list, raw_signal_plot_collection_patch, figure_shapes = uheat.move_score_shape(session_id, folder_name, signal_names, figure_shapes, raw_signal_figure_ids, relayout_data)
    else:
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {ctx.triggered_id}. No processing function known for {relayout_data=}.")
        raise PreventUpdate

    return figure_shape_patch, raw_signal_plot_patch_list, raw_signal_plot_collection_patch, figure_shapes


@callback(
    Output({"type": on_click_type, "index": ALL}, "figure", allow_duplicate=True),
    Output("heatmap-figure-shape-store", "data", allow_duplicate=True),
    Output({"type": on_click_type, "index": ALL}, "clickData"),
    Input({"type": on_click_type, "index": ALL}, "clickData"),
    State({"type": on_click_type, "index": ALL}, "id"),
    State("heatmap-figure-shape-store", "data"),
    State(heatmap_id, "id"),
    prevent_initial_call=True,
)
def add_click_line(click_data, figure_ids: list[str], figure_shapes: dict[str: list], heatmap_fig_id: str):
    return uheat.draw_lines_on_click(click_data, figure_ids, figure_shapes, line_keywords={stringify_id(heatmap_fig_id): {'color': 'white', 'width': 3}})


# delete all shapes upon heatmap change
@callback(
Output(component_id=heatmap_id, component_property='figure', allow_duplicate=True),
    Output(component_id="heatmap-raw-signal-graph-container", component_property='children', allow_duplicate=True),
    Output(component_id="heatmap-figure-shape-store", component_property="data", allow_duplicate=True),
    Input('heatmap-displayed-signal-store', "data"),
    State('heatmap-figure-shape-store', "data"),
    State(heatmap_id, "id"),
    prevent_initial_call=True,
)
def heatmap_clear_patches(displayed_signals: list[str], figure_shapes: dict, heatmap_fig_id: dict):

    # make a patch for the figure
    figure_patch = Patch()
    children_patch = Patch()

    # get the old heatmap range
    datarange, shapes = unpack_shape_store_entry(figure_shapes[stringify_id(heatmap_fig_id)])

    # create a new heatmap shape entry
    new_entry = make_shape_store_entry(*datarange)

    # clear the figure shapes
    figure_shapes.clear()
    figure_shapes[stringify_id(heatmap_fig_id)] = new_entry

    # clear the children
    children_patch.clear()

    # clear the figure shapes
    figure_patch['layout']['shapes'].clear()

    return figure_patch, children_patch, figure_shapes


"""
@callback(
    Input("heatmap-figure-shape-store", "data"),
)
def print_thing(data):
    print('-------------- START ----------------')
    for key, val in data.items():
        print(key)
        for shape in val['shapes']:
            print(shape['type'], shape['x0'], shape['x1'])
        print()
    print('--------------- END -----------------')
    print()


@callback(
    Input("heatmap-active-signal-store", "data"),
)
def print_thing(data):
    print(data)
  """
