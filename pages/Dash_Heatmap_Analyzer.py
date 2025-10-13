import logging
import time
import functools

import plotly.graph_objs
from dash import dcc, html, Input, Output, State, ctx, ClientsideFunction, register_page, clientside_callback, callback
import dash
from dash_extensions import EventListener
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

import util.load_data as utl
import util.prepocessing as prep
from GLOBALS import *
import util.cache_registry as ucache


# register the page to our application
register_page(__name__, path="/heatmap")
logger = logging.getLogger("frontend-logger")

# define some styles for different html elements
dash_heatmap_styles = {
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


@ucache.lru_cache(1)
def process_signals(session_id: str, folder_name: str, window_size: str = None, signal_list: tuple[str] = None, normalization_window_size: int = None) -> tuple[pd.DataFrame, int, tuple[int]]:
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
    score_df = functools.reduce(lambda left, right: pd.merge(left, right, on='timestamp', how='outer'), scores.values())

    # normalize the score
    score_df = prep.normalization(score_df, window_length=normalization_window_size)

    # make the reference signal subtraction
    score_df = score_df.sub(score_df.mean(axis=1), axis="index").abs()
    # TODO: do not absolute value the negative scores

    # transform the index into a pandas timestamp
    score_df.index = pd.to_datetime(score_df.index)

    logger.info(f"[{__name__}] Preprocessed data in {time.perf_counter() - start:0.2f} s. Parameters: {window_size=}, {signal_list=}, {normalization_window_size=}")
    return score_df, window_size, window_sizes


def create_heatmap(session_id: str, folder_name: str, window_size: int = None, signal_list: tuple[str] = None, normalization_window_size: int = None) -> plotly.graph_objs.Figure:
    # read the signals
    score_df, _, _ = process_signals(session_id=session_id, folder_name=folder_name, window_size=window_size, signal_list=signal_list, normalization_window_size=normalization_window_size)

    # make the figure
    fig = draw_heatmap(score_df)
    return fig


@ucache.lru_cache(1)
def get_explanation():
    # get the explanation text from the file
    with open("./assets/explanation.txt") as filet:
        text = filet.read()
    return text


# dash app layout ------------------------------------------------------------------------------------------------------

# how to call an update from button presses modified from
# https://community.plotly.com/t/how-to-get-trigger-from-keyboard/76960/6
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="delete_stuff"),
    Output("heatmap-signal-graph", "id"),
    Input("heatmap-signal-graph", "id"),
)

# how to make draggable signals
# https://community.plotly.com/t/ordering-children-with-drag-drop-how-to-trigger-dash-component-has-changed-from-js/47243/10
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="make_draggable"),
    Output("drag-container", "data-drag"),
    [Input("drag-container", "id"),
     Input("drag-ignore-container", "id")],
)

# textbox selection with regex
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="select_signals"),
    Output("signal-include-container", "children"),
    [Input("signal-include-text-search", "value"),
     Input("drag-container", "id"),
     Input("drag-ignore-container", "id"),
     Input("drag-container", "id")
     ],
)
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="select_signals"),
    Output("signal-ignore-container", "children"),
    [Input("signal-ignore-text-search", "value"),
     Input("drag-ignore-container", "id"),
     Input("drag-container", "id"),
     Input("drag-container", "id")
     ],
)

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
    figure = create_heatmap(session_id=session_id, folder_name=folder_name, window_size=window_size)
    logger.info(f"[{__name__}] Created initial figure and window size options.")

    # define an event for our custom event listener to drag and drop buttons
    drop_event = {"event": "dropcomplete", "props": ["detail.name", "detail.children"]}
    delete_event = {"event": 'shapeDeletion', "props": ["detail.children"]}

    # get the explanation text from the file
    text = get_explanation()

    # define the app layout
    layout_definition = (
        html.Div([
            html.Div(children=[
                html.H1(f'Change Decoupling Heatmap Analyzer',
                        style={'fontSize': 40},
                        id='header'),
                dbc.Accordion(children=[
                    dbc.AccordionItem(children=[html.Pre(text, className='pre-expl')], title="Explanation", )
                ],
                    start_collapsed=True,
                ),
                "Window Size Selection"
                ,
                dbc.Select(id="select-window-size", options=sorted(window_sizes), value=window_size),
            ],
                style=dash_heatmap_styles['div'],
            ),
            html.Div(children=[
                html.H3("Signal Selection"),
                dbc.Accordion(children=[
                    dbc.AccordionItem(children=[
                        EventListener(children=[
                            dbc.Row(children=[
                                dbc.Col(children=[
                                    'Ignore signals with pattern: ',
                                    dcc.Input(
                                        id="signal-include-text-search",
                                        type='text',
                                        placeholder="Signal Regex",
                                        debounce=True,
                                    ),
                                ]),
                                dbc.Col(children=[
                                    'Plot signals with pattern: ',
                                    dcc.Input(
                                        id="signal-ignore-text-search",
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
                                        dbc.CardBody(children=[
                                            html.Div(col, id=f"signal-{col}", disable_n_clicks=True,
                                                                className='a')
                                                       for col in score_df.columns],
                                            id='drag-container', style={"min-height": "10px"}),
                                            ]),
                                        ]),
                                dbc.Col(children=[
                                    dbc.Card(children=[
                                        dbc.CardHeader(["Ignored Columns"]),
                                        dbc.CardBody(children=[
                                            html.Div(children=[], id='drag-ignore-container', style={"min-height": "10px"})
                                        ]),
                                    ]),
                                ]),
                            ]),
                        ],
                            events=[drop_event], logging=False, id="el",
                        ),
                        html.Br(),
                        dcc.Input(
                            id="normalization-input",
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
            ],
                style=dash_heatmap_styles['div'],
                id='column-container'
            ),
            html.Div(children=[
                "Shortcuts: [DEL] deletes the last drawn rectangle. [0] sets view to anomaly_heatmap. Any other number scrolls to "
                "the regarding selection, e.g. [1].",
                EventListener(children=[
                    html.Div(children=[
                        dcc.Graph(
                            id='heatmap-signal-graph',
                            figure=figure,
                        ),
                    ]),
                ],
                    events=[delete_event], logging=True, id="delete-listener",
                ),
                html.Button("Delete All", id="delete-button-all", className="button-4"),
            ],
                style=dash_heatmap_styles['stickydiv'] | dash_heatmap_styles['div']
            ),
            html.Div(children=[
                ],
                    id="overall-div"
                ),
            html.Div(
                children=[],
                style=dash_heatmap_styles['div'],
                id='text-container',
                hidden=True,
            ),
            html.Div(
                children=[],
                style=dash_heatmap_styles['div'],
                id='signal-include-container',
                hidden=True,
            ),
            html.Div(
                children=[],
                style=dash_heatmap_styles['div'],
                id='signal-ignore-container',
                hidden=True,
            ),
        ],
        ))
    return layout_definition


# app callbacks --------------------------------------------------------------------------------------------------------


@callback(
    Output(component_id='heatmap-signal-graph', component_property='figure', allow_duplicate=True),
    Output(component_id='overall-div', component_property='children', allow_duplicate=True),
    Output(component_id='delete-button-all', component_property='children'),
    Output("el", "event"),
    Output("delete-listener", "event"),
    Input(component_id='delete-button-all', component_property='n_clicks'),
    State('heatmap-signal-graph', 'figure'),
    Input('heatmap-signal-graph', 'relayoutData'),
    State(component_id='overall-div', component_property='children'),
    Input("el", "n_events"),
    State("el", "event"),
    State(component_id="drag-container", component_property="children"),
    Input("delete-listener", "n_events"),
    State("delete-listener", "event"),
    Input("normalization-input", "value"),
    State("session-id", "data"),
    State("folder-name", "data"),
    Input("select-window-size", "value"),
    prevent_initial_call=True,
)
def delete_shapes(delete_all, current_figure, relayoutData, all_children,
                  nevents, event_data, children,
                  n_delete_events, delete_event_data, normalization_size,
                  session_id, folder_name, window_size):

    # check whether the relayoutData was the trigger, but it was only sizing
    if ctx.triggered_id == 'heatmap-signal-graph' and relayoutData == {'autosize': True}:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    logger.info(f"[{__name__}] Triggered Element {ctx.triggered_id}, {ctx.inputs}")

    # get the new columns
    if event_data is not None:
        new_cols = [ele.split('-')[1] for ele in event_data['detail.children']]
    else:
        new_cols = current_figure['data'][0]['y']

    # get the original data
    if normalization_size is not None:
        if normalization_size == 0:
            normalization_size = None
        else:
            normalization_size = abs(normalization_size)
    score_df, _, _ = process_signals(session_id, folder_name, window_size=window_size, signal_list=tuple(new_cols),
                                     normalization_window_size=normalization_size)

    # get existing shapes and sort out all that are not rect
    if len(current_figure['data'][0]['y']) == len(new_cols):
        shapes = [ele for ele in current_figure['layout'].get('shapes', []) if ele['type'] == 'rect']

        # delete the shapes so we can update them later
        if 'shapes' in current_figure['layout']:
            current_figure['layout']['shapes'].clear()
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
    if 'delete-button-all' == ctx.triggered_id:
        logger.info(f"[{__name__}] Delete all shapes {ctx.triggered_id}, {ctx.inputs}")
        shapes.clear()

    # update the shape accordingly (by snapping to complete integers) and only keep shapes that we like
    for idx, shape in enumerate(shapes):

        # set the indices
        shape['y0'] = float(round(shape['y0']))
        shape['y1'] = float(round(shape['y1']))

        # give the shape a text
        shape['label']['text'] = f'Selection {idx + 1}'
        shape['label']['font'] = {'color': 'white'}

        # check whether we have to delete the texttemplate
        if 'texttemplate' in shape['label']:
            del shape['label']['texttemplate']

    # get the signals
    _, _, _, _, _, raw_signals_grouped, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # make the other plots for each of the shapes
    all_children.clear()
    for idx, shape in enumerate(shapes):

        # get all the columns we need to select (depending on where we start, one of the shapes is the larger one)
        column_start = round(shape['y0'])
        column_end = round(shape['y1'])
        column_start, column_end = sorted((column_start, column_end))
        column_start = max(column_start, 0)
        column_end = min(column_end, len(score_df.columns)-1)

        # get the time we need to select (depending on where we start, one of the shapes is the larger one)
        time_start = shape['x0']
        time_end = shape['x1']
        start_idx, end_idx = sorted((time_start, time_end), key= lambda x: pd.Timestamp(x))
        # start_idx, end_idx = sorted([pd.Timestamp(time_start), pd.Timestamp(time_end)])
        # create the new dataframe
        df = pd.concat((raw_signals_grouped.get_group(group).loc[start_idx:end_idx, ['normalized value', 'sensor']] for group in score_df.columns[column_start:column_end]))

        # make a figure and plot all the signals
        fig = px.line(df, x=df.index, y='normalized value', color='sensor')
        fig.update_layout(xaxis_title='Time', yaxis_title='Signal Value (MinMax normalized)')

        all_children.append(html.Div(children=[html.H3(f"Signals for Selection {idx+1}"),
                                               html.Details(children=[
                                                                html.Div(children=[
                                                                    dcc.Graph(figure=fig)])
                                                            ],
                                                            open=True,
                                                            id=f'signal-selection{idx + 1}'
                                                            )]
                                     , style=dash_heatmap_styles['div'], id=f'heatmap-signal-selection-div-{idx+1}'
                                     )
                            )

    # only redraw the figure if we changed some of the columns or deleted s
    if ctx.triggered_id == "el":
        current_figure = draw_heatmap(score_df[new_cols])
    else:
        current_figure = plotly.graph_objs.Figure(current_figure, )

    # update figure layout (shapes and also the ordering on the yaxis)
    current_figure.update_layout(shapes=shapes)

    return current_figure, all_children, f"Delete all {len(shapes)}", None, None


@callback(
    Output('overall-div', 'children'),
    Input('heatmap-signal-graph', 'clickData'),
    State('overall-div', 'children'),
    prevent_initial_call=True,
)
def click_into_heatmap(click_data: dict, all_children):

    if click_data is None:
        return dash.no_update, dash.no_update

    # get the signals we selected
    selected_time = [point["x"] for point in click_data["points"]][0]

    # go through all the children with the signal data
    for child in all_children:
        # get the figure from the nested structure
        fig = child['props']['children'][1]['props']['children'][0]['props']['children'][0]['props']['figure']

        # get the data from within the figure
        data = fig['data'][0]['x']
        st, en = data[0], data[-1]

        # delete old shapes
        fig['layout']['shapes'] = []

        # convert into figure
        fig = plotly.graph_objs.Figure(fig)

        if pd.Timestamp(st) <= pd.Timestamp(selected_time) <= pd.Timestamp(en):
            fig.add_vline(x=selected_time)
        child['props']['children'][1]['props']['children'][0]['props']['children'][0]['props']['figure'] = fig

    return all_children


@callback(Output('text-container', 'children'), Input('heatmap-signal-graph', 'relayoutData'), State('heatmap-signal-graph', 'figure'), prevent_initial_call=True)
def test_shapes(relayoutData, figure):
    if "shapes" not in relayoutData:
        return "AWAY"
    return "DEBUG"
