import re
import os
import inspect
import logging
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash

import util.load_data as utl
import util.styles as usty
from GLOBALS import *


def draw_heatmap(data: pd.DataFrame):

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
    fig.update_layout(shapes=[])  # important otherwise our patches to the shape property won't work

    return fig


def make_selection_title(idx: int):
    return f'Selection {idx}'

def make_raw_signal_plot_title(idx: int):
    return f'Signal for {make_selection_title(idx)}'

def is_custom_shape(shape: dict):
    return shape['type'] == 'rect' or shape['type'] == 'path'

def figure_get_line(shapes_list: list[dict]):
    line_idx, line_shape = next(((idx, shape) for idx, shape in enumerate(shapes_list) if shape['type'] == 'line'), (None, None))
    return line_idx, line_shape

def get_custom_shapes(shapes_list: list[dict]):
    return [(idx, ele) for idx, ele in enumerate(shapes_list) if is_custom_shape(ele)]

def make_shape_store_entry(xmin: str, xmax: str):
    return {'range': (xmin, xmax), 'shapes': []}


def make_vline(x_position: [float | pd.Timestamp], width: float = 2, color: str = "black", dash_type: str = "solid", layer: str = "below"):

    line = {
        "type": "line",
        "xref": "x",
        "yref": "paper",  # span full plotting height
        "x0": x_position,
        "x1": x_position,
        "y0": 0,
        "y1": 1,
        "layer": layer,
        "line": {
            "color": color,
            "width": width,
            "dash": dash_type,
        },
    }
    return line


def shape_update_patch(shape: dict, figure_shape_patch: dash.Patch, shape_idx: int, title_idx: int = None):

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

    # disable editing
    # figure_shape_patch['layout']['shapes'][shape_idx]['editable'] = False

    # give the shape a text if it is a new shape
    if title_idx is not None:
        title = make_selection_title(title_idx)
        figure_shape_patch['layout']['shapes'][shape_idx]['label']['text'] = title
        shape['label']['text'] = title
        figure_shape_patch['layout']['shapes'][shape_idx]['label']['font'] = {'color': 'white'}
        shape['label']['font'] = {'color': 'white'}
        figure_shape_patch['layout']['shapes'][shape_idx]['label']['textposition'] = 'bottom center'
        shape['label']['textposition'] = 'bottom center'

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
    fig.update_layout(shapes=[]) # important otherwise our patches to the shape property won't work

    return fig, names, time_start, time_end


def create_empty_figure_with_text(text: str):
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        text=text,
        showarrow=False,
        font=dict(size=24),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def create_new_raw_signal_plot(session_id: str, folder_name: str, signal_names: list[str], figure_shapes: dict[str:list], raw_signal_figure_ids: list[str], relayout_data: dict, signal_graph_type: str):

    # get the logger
    logger = logging.getLogger("frontend-logger")

    # check whether our shapes are really up to date
    trigger_id = dash.stringify_id(dash.ctx.triggered_id)
    all_shapes = figure_shapes.get(trigger_id, {'shapes': []})['shapes']
    if any(saved_shape['type'] != shape['type'] for saved_shape, shape in zip(all_shapes, relayout_data['shapes'])):
        logger.warning(f"[{__name__}][{inspect.stack()[0][3]}] Something with the shape store is off.")

    # get existing rectangle shapes
    shapes = get_custom_shapes(relayout_data['shapes'])

    # if there are no rectangle shapes we do not have to do anything
    if not shapes:
        raise dash.exceptions.PreventUpdate

    # create our patch objects for the raw signal plots
    raw_signal_plot_collection_patch = dash.Patch()
    figure_shape_patch = dash.Patch()

    # get the latest shape and its number
    idx = len(shapes)
    shape_dx, shape = shapes[idx-1]

    # update the shape
    shape_y0, shape_y1 = shape_update_patch(shape, figure_shape_patch, shape_dx, idx)

    # get the current time as an into to grant unique ids
    currtime = str(time.time_ns())

    # get the timestamps
    time_start, time_end = pd.Timestamp(0), pd.Timestamp(0)

    # check if we have too many shapes already
    if len(shapes) > MAX_PLOTLY_SHAPES-3:
        text_notification = '<a href="https://plotly.com/python/performance/">Too many shapes</a>. Please delete some. Otherwise, rendering will fail.'
        fig = create_empty_figure_with_text(str(text_notification))
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Too many shapes: {len(shapes)=}.")

    # check if we want to draw too many shapes
    elif abs(shape_y1-shape_y0)+1 > RAW_SIGNAL_PLOT_MAXIMUM_NUMBER:
        text_notification = "Too many signals in this selection"
        fig = create_empty_figure_with_text(str(text_notification))
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Too many signals: ({shape_y0=}, {shape_y1=}).")

    # draw the figure
    else:
        # make the figure
        fig, names, time_start, time_end = create_raw_signal_figure(session_id, folder_name, shape, shape_y0, shape_y1, signal_names)

    # create graph object
    graph_id = {"type": signal_graph_type, "index": currtime}
    stringified_graph_id = dash.stringify_id(graph_id)
    figure_shapes[stringified_graph_id] = make_shape_store_entry(*map(str, (time_start, time_end)))
    raw_signal_graph = dash.dcc.Graph(figure=fig, id=graph_id)

    # get the line from the heatmap figure
    _, line_obj = figure_get_line(figure_shapes[trigger_id]['shapes'])
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
    new_raw_plot = dash.html.Div(children=[dash.html.H3(make_raw_signal_plot_title(idx)),
                                      dash.html.Details(children=[
                                          dash.dcc.Loading(children=[
                                              dash.html.Div(children=[
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

    # get the logger
    logger = logging.getLogger("frontend-logger")

    # write to logger
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Update shapes!")

    # create our patch objects for our figure
    figure_shape_patch = dash.Patch()
    graph_patch = dash.Patch()
    raw_container_patch = dash.no_update

    # extract the current shape information
    pattern = re.compile(r'\[(\d+)]\.(\w+)$')
    elements_dict = {(int(n), term): value for (n, term), value in
                     ((re.search(pattern, key).groups(), value) for key, value in relayout_data.items())}

    # check that the shape index is unique
    shape_index = set(ele[0] for ele in elements_dict.keys())

    # check our extraction results
    if len(shape_index) != 1:
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {dash.ctx.triggered_id}. We did not find relocate variables in {relayout_data=}.")
        raise dash.exceptions.PreventUpdate

    # get the index
    shape_dx = shape_index.pop()

    # get the shapes
    trigger_id = dash.stringify_id(dash.ctx.triggered_id)
    shapes_idces = [idx for idx, ele in get_custom_shapes(figure_shapes[trigger_id]['shapes'])]

    # find the div idx
    div_dx = shapes_idces.index(shape_dx)
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Moved shape {shape_dx} with corresponding div {div_dx}.")

    # build a dummy shape with all the necessary information
    shape_update = {key[1]: val for key, val in elements_dict.items()}

    # update the shape itself
    shape = figure_shapes[trigger_id]['shapes'][shape_dx]
    shape.update(shape_update)

    # update the shape
    shape_y0, shape_y1 = shape_update_patch(shape, figure_shape_patch, shape_dx, None)

    # check whether we have too many shapes
    if div_dx + 1 > MAX_PLOTLY_SHAPES - 3:
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Too many shapes. Ignore redraw of plot {div_dx}.")
        raise dash.exceptions.PreventUpdate

    # get the id of the figure we want change (offset by one since the first is the heatmap)
    target_raw_signal_figure_id = raw_signal_figure_ids[div_dx+1]

    # check whether we want to draw too many signals per shape
    if abs(shape_y1 - shape_y0) + 1 > RAW_SIGNAL_PLOT_MAXIMUM_NUMBER:
        logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Too many signals: ({shape_y0=}, {shape_y1=}).")
        fig = create_empty_figure_with_text("Too many signals in this selection")
        time_start  = pd.Timestamp(1)
        time_end = pd.Timestamp(0)
    # create the figure from the newly selected data
    else:
        fig, _, time_start, time_end = create_raw_signal_figure(session_id, folder_name, shape, shape_y0, shape_y1, signal_names)

    # embed figure into a dcc graph
    graph = dash.dcc.Graph(figure=fig, id=target_raw_signal_figure_id)

    # get the line from the heatmap figure
    _, line_obj = figure_get_line(figure_shapes[trigger_id]['shapes'])
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
    figure_shapes[dash.stringify_id(target_raw_signal_figure_id)]['shapes'] = new_shape_list
    figure_shapes[dash.stringify_id(target_raw_signal_figure_id)]['range'] = (time_start, time_end)

    # check whether the line exists and is in range

    # create the update list (one less than our ids, since we do not patch the heatmap)
    graph_patch[0] = graph
    update_list = [graph_patch if idx == div_dx else dash.no_update for idx in range(len(raw_signal_figure_ids)-1)]

    # update the shape dict
    figure_shapes[trigger_id][shape_dx] = shape

    return figure_shape_patch, update_list, raw_container_patch, figure_shapes


def draw_lines_on_click(click_data, figure_ids: list[str], figure_shapes: dict[str: list], line_keywords: [dict[str:] | None] = None):

    # make the default dict
    if line_keywords is None:
        line_keywords = dict()

    # get the active click data
    active_click_idx, active_click = next(((idx, x) for idx, x in enumerate(click_data) if x is not None), (None, None))

    # check whether we clicked into any plot
    if active_click is None:
        raise dash.exceptions.PreventUpdate

    # get the x position of the click
    clicked_x = pd.Timestamp(active_click['points'][0]['x'])

    # check whether we have a close line
    close_to_active = False

    # check whether the clicked figure has a line
    clicked_fig_id = dash.stringify_id(figure_ids[active_click_idx])
    line_idx, line_shape = figure_get_line(figure_shapes[clicked_fig_id]['shapes'])

    # get the current position of the line if there is one
    if line_idx is not None:

        # get the range of the current active figure
        fig_range = sorted(map(pd.Timestamp, figure_shapes[clicked_fig_id]['range']))

        # get the position of the line
        current_position = pd.Timestamp(figure_shapes[clicked_fig_id]['shapes'][line_idx]['x0'])

        # check whether we are close to active
        if abs(current_position - clicked_x) < 0.05 * abs(fig_range[0] - fig_range[1]):
            close_to_active = True

    # update all the shapes of all figures
    figure_patches = []
    for figid in figure_ids:

        # get the string id
        figid = dash.stringify_id(figid)

        # create the patch
        figure_patch = dash.Patch()

        # make the vline
        vline = make_vline(clicked_x, **line_keywords.get(figid, dict()))

        # check whether we already have a line shape close by
        line_idx, line_shape = figure_get_line(figure_shapes[figid]['shapes'])

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

    return figure_patches, figure_shapes, [None] * len(figure_ids)


def delete_shapes(delete_event_data: dict, raw_signal_fig_ids: list, figure_shapes: dict[str: list], heatmap_id: dict):

    # get the logger
    logger = logging.getLogger("frontend-logger")

    if delete_event_data is None:
        raise dash.exceptions.PreventUpdate

    # check whether there are existing rectangle shapes
    stringy_heatmap_id = dash.stringify_id(heatmap_id)
    shape_idces = [idx for idx, ele in get_custom_shapes(figure_shapes[stringy_heatmap_id]['shapes'])]
    if not shape_idces:
        raise dash.exceptions.PreventUpdate

    # create our patch objects for our figure
    figure_shape_patch = dash.Patch()
    scatter_overall_patch = dash.Patch()

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
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Triggered Element {dash.ctx.triggered_id}. We want to delete!: {active_idx=}.")

    # update the remaining shape names and the headers of the remaining raw signal plots
    new_index = 1
    for idx in shape_idces:

        # ignore the divs that will be deleted
        if idx in active_idx:
            continue

        # adapt the text of the selections
        figure_shape_patch['layout']['shapes'][idx]['label']['text'] = make_selection_title(new_index)

        # get the index of the div (this accounts for the line that we use as marker)
        div_idx = shape_idces.index(idx)

        # update the corresponding raw signal plot headers
        scatter_overall_patch[div_idx]["props"]["children"][0]["props"]["children"] = make_raw_signal_plot_title(new_index)

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
        del figure_shapes[dash.stringify_id(raw_signal_fig_ids[div_idx+1])]

    return figure_shape_patch, scatter_overall_patch, figure_shapes