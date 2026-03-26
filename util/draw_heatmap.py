import re
import pandas as pd
import plotly.express as px
import os

import dash
import util.load_data as utl
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
    line_idx, line_shape = next(((idx, shape) for idx, shape in enumerate(shapes_list) if shape['type'] == 'line'),(None, None))
    return line_idx, line_shape

def get_custom_shapes(shapes_list: list[dict]):
    return [(idx, ele) for idx, ele in enumerate(shapes_list) if is_custom_shape(ele)]


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