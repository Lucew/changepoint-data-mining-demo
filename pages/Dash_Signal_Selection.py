import logging
import time
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

import util.load_data as utl
import util.prepocessing as prep
import util.residuals as procd
import util.styles as usty
import util.create_tsne as utsne
import util.draw_heatmap as uheat
import util.draw_scatter as uscat
from GLOBALS import *


# register the page to our application
register_page(__name__, path="/scatter")
logger = logging.getLogger("frontend-logger")

# set an id for the score graph
score_graph_type = "raw-signal-graph"
score_graph_index = "scatter-signal-graph"
score_graph_id = {"type": score_graph_type, "index": score_graph_index}


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


def make_shape_store_entry(xmin: str, xmax: str):
    return {'range': (xmin, xmax), 'shapes': []}


# app callbacks --------------------------------------------------------------------------------------------------------

@callback(
    Output(score_graph_id, component_property="figure", allow_duplicate=True),
    Output(component_id='div-scatter-signal-graph', component_property="hidden"),
    Output(component_id='scatter-overall-div', component_property='hidden'),
    Output(component_id='scatter-overall-div2', component_property='hidden', allow_duplicate=True),
    Output(component_id='div-scatter-signal-graph2', component_property='hidden', allow_duplicate=True),
    Output("graph-loading-signal", "children", allow_duplicate=True),
    Output("signal-name-store", "data"),
    Output(component_id='scatter-overall-div', component_property='children'),
    Output("figure-shape-store", "data", allow_duplicate=True),
    State("session-id", "data"),
    State("folder-name", "data"),
    State(score_graph_id, component_property="id"),
    Input(component_id="scatter-graph", component_property="selectedData"),
    prevent_initial_call=True)
def select_signals_scatter(session_id: str, folder_name: str, graph_id, selected_data):

    # check whether we selected nothing
    children_patch = Patch()
    children_patch.clear()
    if selected_data is None or len(selected_data["points"]) == 0:
        return go.Figure(go.Scatter(x=[], y=[])), True, True, dash.no_update, dash.no_update, "", [], children_patch, dict()

    # get the signals we selected
    selected_signals = [point["customdata"][0] for point in selected_data["points"]]

    # get the regression results
    regression_results, _, _ = prep.preprocess_regression_results(session_id, folder_name)

    # get the scores from the files
    scores, _, _, _, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # make a logger entry
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Requested heatmap for {len(selected_signals)} signals.")

    # compute the weighted scoring for each of the selected signals
    # TODO mean correlation value per cluster as hint whether it is a good cluster
    result_df = procd.compute_weighted_residual_norm(regression_results, selected_signals, scores, coming_from='signal-selection')

    # normalize the signals
    result_df = prep.normalization(result_df)

    # make the figure from the signals
    fig = uheat.draw_heatmap(result_df)
    shape_dict = {stringify_id(graph_id): make_shape_store_entry(str(result_df.index.min()), str(result_df.index.max()))}

    return fig, False, False, True, True, "", list(result_df.columns), children_patch, shape_dict


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


def make_vline(x_position: [float | pd.Timestamp], width: float = 2, color: str = "black", dash_type: str = "solid"):

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
    return (uscat.create_scatter(session_id, folder_name, value, correlation_threshold, component_list, measurement_list),
            uscat.create_scatter_3d(session_id, folder_name,value, correlation_threshold, component_list, measurement_list))


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


@callback(
    Output(component_id=score_graph_id, component_property='figure', allow_duplicate=True),
    Output(component_id="scatter-delete-listener", component_property="event", allow_duplicate=True),
    Output(component_id="scatter-overall-div", component_property="children", allow_duplicate=True),
    Output("figure-shape-store", "data", allow_duplicate=True),
    Input(component_id="scatter-delete-listener", component_property="n_events"),
    State(component_id="scatter-delete-listener", component_property="event"),
    State(component_id={"type": "raw-signal-graph", "index": ALL}, component_property='id'),
    State("figure-shape-store", "data"),
    State(component_id=score_graph_id, component_property='id'),
    prevent_initial_call=True
)
def delete_shapes(n_delete_events, delete_event_data, raw_signal_fig_ids: list, figure_shapes: dict[str: list], heatmap_id: dict):

    if delete_event_data is None:
        raise PreventUpdate

    # check whether there are existing rectangle shapes
    stringy_heatmap_id = stringify_id(heatmap_id)
    shape_idces = [idx for idx, ele in uheat.get_custom_shapes(figure_shapes[stringy_heatmap_id]['shapes'])]
    if not shape_idces:
        raise PreventUpdate

    # create our patch objects for our figure
    figure_shape_patch = Patch()
    scatter_overall_patch = Patch()

    # get the index of the active shape
    active_idx = delete_event_data['detail.children']

    # get the indices we want to delete
    if not active_idx:
        active_idx = [shape_idces[-1]]
    elif active_idx == 'all':
        active_idx = shape_idces
    else:
        active_idx = list(map(int, active_idx))
    active_idx = set(active_idx)

    # write to logger
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {ctx.triggered_id}. We want to delete!: {active_idx=}.")

    # update the remaining shape names and the headers of the remaining raw signal plots
    new_index = 1
    for idx in shape_idces:

        # ignore the divs that will be deleted
        if idx in active_idx:
            continue

        # adapt the text of the selections
        figure_shape_patch['layout']['shapes'][idx]['label']['text'] = uheat.make_selection_title(new_index)

        # get the index of the div (this accounts for the line that we use as marker)
        div_idx = shape_idces.index(idx)

        # update the corresponding raw signal plot headers
        scatter_overall_patch[div_idx]["props"]["children"][0]["props"]["children"] = uheat.make_raw_signal_plot_title(new_index)

        # increment the new index
        new_index += 1

    # delete the shapes in the figure
    # !IMPORTANT! Do this as last step. Otherwise, shape indices will be incorrect.
    # !IMPORTANT! Do this in reverse index order

    for ele in sorted(active_idx, reverse=True):

        # delete the target shapes from the heatmap
        del figure_shape_patch['layout']['shapes'][ele]

        # update the heatmap shapes in our shape store
        del figure_shapes[stringy_heatmap_id]['shapes'][ele]

        # get the index of the div (this accounts for the line that we use as marker)
        div_idx = shape_idces.index(ele)

        # delete the raw signal plots
        del scatter_overall_patch[div_idx]

        # update our shape store by deleting the key corresponding to the raw signal plot that we deleted
        # increment index by one as the first id in raw_signal_fig_ids is the heatmap id
        del figure_shapes[stringify_id(raw_signal_fig_ids[div_idx+1])]

    return figure_shape_patch, None, scatter_overall_patch, figure_shapes


@callback(
    Output(component_id=score_graph_id, component_property='figure', allow_duplicate=True),
    Output(component_id={"type": "div-raw-signal-graph", "index": ALL}, component_property='children', allow_duplicate=True),
    Output(component_id='scatter-overall-div', component_property='children', allow_duplicate=True),
    Output("figure-shape-store", "data", allow_duplicate=True),
    State("session-id", "data"),
    State("folder-name", "data"),
    State("signal-name-store", "data"),
    State("figure-shape-store", "data"),
    State(component_id={"type": "raw-signal-graph", "index": ALL}, component_property='id'),
    State(component_id={"type": "div-raw-signal-graph", "index": ALL}, component_property='id'),
    Input(score_graph_id, "relayoutData"),
    running=[(Output("loading-signals", "display"), "show", "auto")], # this deactivates the figure while running our function
    prevent_initial_call=True
)
def modify_shapes(session_id: str, folder_name: str, signal_names: list[str], figure_shapes: dict[str:list], raw_signal_figure_ids: list[str], raw_signal_div_ids: list[str], relayout_data: dict):

    # check whether we want to draw a new shape
    if 'autosize' in relayout_data:
        raise PreventUpdate

    is_new_shape = 'shapes' in relayout_data
    is_shape_redraw = any(key.startswith('shapes[') for key in relayout_data.keys())
    if not is_new_shape and not is_shape_redraw:
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {ctx.triggered_id}. Unknown event type: {relayout_data=}.")
        raise PreventUpdate

    # make a logging entry
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {ctx.triggered_id}, {is_new_shape=}, {is_shape_redraw=}.")

    # make the processing depending on the event
    if is_new_shape:
        figure_shape_patch, raw_signal_plot_patch_list, raw_signal_plot_collection_patch, figure_shapes = create_new_raw_signal_plot(session_id, folder_name, signal_names, figure_shapes, raw_signal_figure_ids, relayout_data)
    elif is_shape_redraw:
        figure_shape_patch, raw_signal_plot_patch_list, raw_signal_plot_collection_patch, figure_shapes = move_score_shape(session_id, folder_name, signal_names, figure_shapes, raw_signal_figure_ids, relayout_data)
    else:
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {ctx.triggered_id}. No processing function known for {relayout_data=}.")
        raise PreventUpdate

    return figure_shape_patch, raw_signal_plot_patch_list, raw_signal_plot_collection_patch, figure_shapes


def create_new_raw_signal_plot(session_id: str, folder_name: str, signal_names: list[str], figure_shapes: dict[str:list], raw_signal_figure_ids: list[str], relayout_data: dict):

    # check whether our shapes are really up to date
    trigger_id = stringify_id(ctx.triggered_id)
    all_shapes = figure_shapes.get(trigger_id, {'shapes': []})['shapes']
    if any(saved_shape['type'] != shape['type'] for saved_shape, shape in zip(all_shapes, relayout_data['shapes'])):
        logger.warning(f"[{__name__}][{inspect.stack()[0][3]}] Something with the shape store is off.")

    # get existing rectangle shapes
    shapes = uheat.get_custom_shapes(relayout_data['shapes'])

    # if there are no rectangle shapes we do not have to do anything
    if not shapes:
        raise PreventUpdate

    # create our patch objects for the raw signal plots
    raw_signal_plot_collection_patch = Patch()
    figure_shape_patch = Patch()

    # get the latest shape and its number
    idx = len(shapes)
    shape_dx, shape = shapes[idx-1]

    # update the shape
    shape_y0, shape_y1 = uheat.shape_update_patch(shape, figure_shape_patch, shape_dx, idx)

    # get the current time as an into to grant unique ids
    currtime = str(time.time_ns())

    # create the figure
    fig = None
    text_notification = None
    time_start, time_end = pd.Timestamp(0), pd.Timestamp(0)
    if len(shapes) > MAX_PLOTLY_SHAPES-3:
        text_notification = html.Div([html.A("Too many shapes", href='https://plotly.com/python/performance/'), ". Please delete some. Otherwise, rendering will fail."])
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Too many shapes: {len(shapes)=}.")
    else:
        # make the figure
        fig, names, time_start, time_end = uheat.create_raw_signal_figure(session_id, folder_name, shape, shape_y0, shape_y1, signal_names)

    # create graph object
    graph_id = {"type": "raw-signal-graph", "index": currtime}
    stringified_graph_id = stringify_id(graph_id)
    figure_shapes[stringified_graph_id] = make_shape_store_entry(*map(str, (time_start, time_end)))
    raw_signal_graph = dcc.Graph(figure=fig, id=graph_id)

    # get the line from the heatmap figure
    _, line_obj = uheat.figure_get_line(figure_shapes[trigger_id]['shapes'])
    new_shape_list = []

    if line_obj is not None:
        line_position = pd.Timestamp(line_obj['x0'])

        # check whether there is a line that is in range
        is_in_range = time_start <= line_position <= time_end

        # add the line to the raw signal plot if that is the case
        if is_in_range and fig is not None:
            vline = make_vline(line_position)
            fig.add_shape(vline)
            new_shape_list.append(vline)

    # update the shape store
    figure_shapes[stringified_graph_id]['shapes'] = new_shape_list

    # create the new div
    new_raw_plot = html.Div(children=[html.H3(uheat.make_raw_signal_plot_title(idx)),
                                      text_notification,
                                      html.Details(children=[
                                          dcc.Loading(children=[
                                              html.Div(children=[
                                                  raw_signal_graph,
                                              ],
                                                  id={"type": "div-raw-signal-graph", "index": currtime},
                                              )
                                          ],
                                              overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                                          )
                                      ],
                                          open=True,
                                          id={"type": "raw-signal-div", "index": currtime},
                                      )],
                            style=usty.div_styles['div'],
                            id=f'scatter-signal-selection-div-{currtime}',
                            )

    # append the div to the existing divs
    raw_signal_plot_collection_patch.append(new_raw_plot)

    # update the shape dict
    figure_shapes[trigger_id]['shapes'].append(shape)

    return figure_shape_patch, [dash.no_update]*(len(raw_signal_figure_ids)-1), raw_signal_plot_collection_patch, figure_shapes


def move_score_shape(session_id: str, folder_name: str, signal_names: list[str], figure_shapes: dict[str:list], raw_signal_figure_ids: list[str], relayout_data: dict):

    # write to logger
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Update shapes!")

    # create our patch objects for our figure
    figure_shape_patch = Patch()
    graph_patch = Patch()

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

    # get the shapes
    trigger_id = stringify_id(ctx.triggered_id)
    shapes_idces = [idx for idx, ele in uheat.get_custom_shapes(figure_shapes[trigger_id]['shapes'])]

    # find the div idx
    div_dx = shapes_idces.index(shape_dx)
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Moved shape {shape_dx} with corresponding div {div_dx}.")

    # check whether we have too many shapes
    if div_dx + 1 > MAX_PLOTLY_SHAPES-3:
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Too many shapes. Ignore redraw of plot {div_dx}.")
        raise PreventUpdate

    # build a dummy shape with all the necessary information
    shape_update = {key[1]: val for key, val in elements_dict.items()}

    # update the shape itself
    shape = figure_shapes[trigger_id]['shapes'][shape_dx]
    shape.update(shape_update)

    # update the shape
    shape_y0, shape_y1 = uheat.shape_update_patch(shape, figure_shape_patch, shape_dx, None)

    # get the id of the figure we want change (offset by one since the first is the heatmap)
    target_raw_signal_figure_id = raw_signal_figure_ids[div_dx+1]

    # create the figure and embed it into a dcc graph
    fig, _, time_start, time_end = uheat.create_raw_signal_figure(session_id, folder_name, shape, shape_y0, shape_y1, signal_names)
    graph = dcc.Graph(figure=fig, id=target_raw_signal_figure_id)

    # get the line from the heatmap figure
    _, line_obj = uheat.figure_get_line(figure_shapes[trigger_id]['shapes'])
    new_shape_list = []

    if line_obj is not None:
        line_position = pd.Timestamp(line_obj['x0'])

        # check whether there is a line that is in range
        is_in_range = time_start <= line_position <= time_end

        # add the line to the raw signal plot if that is the case

        if is_in_range:
            vline = make_vline(line_position)
            fig.add_shape(vline)
            new_shape_list.append(vline)

    # update the shapes and the range
    figure_shapes[stringify_id(target_raw_signal_figure_id)]['shapes'] = new_shape_list
    figure_shapes[stringify_id(target_raw_signal_figure_id)]['range'] = (time_start, time_end)

    # check whether the line exists and is in range

    # create the update list (one less than our ids, since we do not patch the heatmap)
    graph_patch[0] = graph
    update_list = [graph_patch if idx == div_dx else dash.no_update for idx in range(len(raw_signal_figure_ids)-1)]

    # update the shape dict
    figure_shapes[trigger_id][shape_dx] = shape

    return figure_shape_patch, update_list, dash.no_update, figure_shapes


@callback(
    Output({"type": "raw-signal-graph", "index": ALL}, "figure", allow_duplicate=True),
    Output("figure-shape-store", "data", allow_duplicate=True),
    Output({"type": "raw-signal-graph", "index": ALL}, "clickData"),
    Input({"type": "raw-signal-graph", "index": ALL}, "clickData"),
    State({"type": "raw-signal-graph", "index": ALL}, "id"),
    State("figure-shape-store", "data"),
    State(score_graph_id, "id"),
    prevent_initial_call=True,
)
def add_click_line(click_data, figure_ids: list[str], figure_shapes: dict[str: list], heatmap_id: str):

    # get the active click data
    active_click_idx, active_click = next(((idx, x) for idx, x in enumerate(click_data) if x is not None), (None, None))

    # check whether we clicked into any plot
    if active_click is None:
        raise PreventUpdate

    # get the x position of the click
    clicked_x = pd.Timestamp(active_click['points'][0]['x'])

    # check whether we have a close line
    close_to_active = False

    # check whether the clicked figure has a line
    clicked_fig_id = stringify_id(figure_ids[active_click_idx])
    line_idx, line_shape = uheat.figure_get_line(figure_shapes[clicked_fig_id]['shapes'])

    # get the current position of the line if there is one
    if line_idx is not None:

        # get the range of the current active figure
        fig_range = sorted(map(pd.Timestamp, figure_shapes[clicked_fig_id]['range']))

        # get the position of the line
        current_position = pd.Timestamp(figure_shapes[clicked_fig_id]['shapes'][line_idx]['x0'])

        # check whether we are close to active
        if abs(current_position - clicked_x) < 0.05*abs(fig_range[0]-fig_range[1]):
            close_to_active = True

    # update all the shapes of all figures
    figure_patches = []
    stringy_id = stringify_id(heatmap_id)
    for figid in figure_ids:

        # get the string id
        figid = stringify_id(figid)

        # create the patch
        figure_patch = Patch()

        # make the vline
        vline = make_vline(clicked_x, color="white" if figid == stringy_id else None, width=4)

        # check whether we already have a line shape close by
        line_idx, line_shape = uheat.figure_get_line(figure_shapes[figid]['shapes'])

        # get the range of the current figure
        fig_range = sorted(map(pd.Timestamp, figure_shapes[figid]['range']))

        # check whether we are in range
        is_in_range = fig_range[0] <= clicked_x <= fig_range[1]

        # if there is no line, we just draw one
        if line_shape is None:

            if is_in_range:
                # create the patch
                figure_patch['layout']['shapes'].append(vline)
                figure_patches.append(figure_patch)

                # update our shape store
                figure_shapes[figid]['shapes'].append(vline)
            else:
                figure_patches.append(dash.no_update)

        # make the patch if it is in range
        else:

            # reposition the line if the click is in range, and we are not close to an existing line
            if is_in_range and not close_to_active:

                # update the line
                figure_patch['layout']['shapes'][line_idx] = vline
                figure_patches.append(figure_patch)

                # update our shape store
                figure_shapes[figid]['shapes'][line_idx] = vline
            else:
                # create the patch
                del figure_patch['layout']['shapes'][line_idx]
                figure_patches.append(figure_patch)

                # update our shape store
                del figure_shapes[figid]['shapes'][line_idx]

    return figure_patches, figure_shapes, [None]*len(figure_ids)


"""@callback(
    Input("figure-shape-store", "data"),
)
def print_thing(data):
    print('-------------- START ----------------')
    for key, val in data.items():
        print(key)
        for shape in val['shapes']:
            print(shape['type'], shape['x0'], shape['x1'])
        print()
    print('--------------- END -----------------')
    print()"""

# how to call an update from button presses modified from
# https://community.plotly.com/t/how-to-get-trigger-from-keyboard/76960/6
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="button_press_interaction"),
    Output(score_graph_id, "id"),
    Input("div-scatter-signal-graph", "id"),
    State('scatter-overall-div', "id"),
)

# delete all shapes
clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="delete_all_shapes"),
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

    # get the result ones before running the app, so we can set some default values
    # variable naming seems complex, but as these are global to the app, these complex names make sure we do not
    # accidentally reuse them
    _global_bokeh_df, _global_default_perplexity, _global_corr_thresh = utsne.create_tsne(session_id, folder_name)
    _global_measurement_types = list(_global_bokeh_df["measurement"].unique())
    _global_component_types = list(_global_bokeh_df["component"].unique())

    # get the explanation text from the file
    with open("./assets/explanation.txt") as filet:
        text = filet.read()

    # define the app layout
    layout_definition = EventListener(children=[html.Div([
        dcc.Store(id="signal-name-store", data=[]),  # to store signal names
        dcc.Store(id="figure-shape-store", data=dict()),  # to store all the shapes we have drawn
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
            style=usty.div_styles['div'],
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
            style=usty.div_styles['div'],
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
                                figure=uscat.create_scatter(session_id, folder_name),
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
                                figure=uscat.create_scatter_3d(session_id, folder_name),
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
            style= usty.div_styles['div'],
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
            style= usty.div_styles['div'] | usty.div_styles['stickydiv'],
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
            style=usty.div_styles['stickydiv'] | usty.div_styles['div'],
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
            style=usty.div_styles['div'],
            hidden=True
        ),
        html.Div(
            children=[],
            style=usty.div_styles['div'],
            id='signal-include-container',
            hidden=True,
        ),
        html.Div(
            children=[],
            style=usty.div_styles['div'],
            id='signal-ignore-container',
            hidden=True,
        ),
    ],
    )
    ],
        events=[delete_event], logging=True, id="scatter-delete-listener", style={"overflow": "clip"},
    )
    return layout_definition