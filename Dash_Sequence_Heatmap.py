import collections
import os.path

from dash import Dash, dcc, html, Input, Output, State, ctx, ClientsideFunction
from dash_extensions import EventListener
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import functools
import glob
from tqdm import tqdm
import multiprocessing as mp
from preprocessing.process import aggregate_ts
import math
import fbpca
import bisect


def normalization(df: pd.DataFrame) -> pd.DataFrame:
    mini = df.min()
    maxi = df.max()
    df -= mini

    # make the normalizer
    norma = maxi - mini
    if isinstance(norma, np.float64):
        if norma == 0:
            norma = 1
    else:
        norma[norma == 0] = 1

    # make the normalization
    df /= norma
    return df


def resample_length(df: pd.DataFrame, num_samples: int) -> pd.DataFrame:

    # Calculate the time range of the original DataFrame
    start_date = df.index.min()
    end_date = df.index.max()

    # Calculate the total duration of the time range
    total_duration = (end_date - start_date).total_seconds()

    # Calculate the number of microseconds per sample
    microseconds_per_sample = math.floor((total_duration / num_samples) * 1e6)

    # Construct the frequency string based on microseconds
    freq_str = f'{microseconds_per_sample}us'

    # make the resampling
    result = {col: aggregate_ts(df.loc[:, [col]], freq_str, emulate_dead_band=False)[col] for col in df.columns}

    # account for small diffs
    for name, series in result.items():
        if series.shape[0] - num_samples > 5:
            f'Something went wrong here. Diff is {series.shape[0] - num_samples}.'
        result[name] = series.iloc[:num_samples]
    df = pd.DataFrame(result)
    return df


def read_file(file: str) -> (int, pd.DataFrame):

    # extract the sequence index from the file
    index = int(os.path.splitext(os.path.split(file)[-1])[0])

    # read the file from the harddrive
    df = pd.read_csv(file, index_col=0, header=0)

    # convert the index to absolute times
    df.index = pd.Timestamp('today').normalize() + pd.to_timedelta(df.index, unit='s')

    return index, df


def process_df(df: pd.DataFrame, target_length: int) -> pd.DataFrame:

    # get all the signals that do not have enough values
    delete = df.isna().sum() > df.shape[0]*0.25
    delete = list(delete[delete].index)

    # drop the columns that do not have enough values
    df = df.drop(delete, axis=1)

    # resample the dataframe to a given length
    df = df.ffill()
    df = df.bfill()
    df = resample_length(df, target_length)

    # normalize the sequences
    df = normalization(df)

    return df


def load_data(number_eigenvectors: int = 5, folder_name: str = "sequences_UnknownSequences") -> (pd.DataFrame, dict[int: pd.DataFrame]):

    # find all the right files
    files = glob.glob(os.path.join("data", folder_name, "*.csv"))

    # sort out the sequence file
    sequencefiles = [file for file in files if not file.endswith('sequences.csv')]
    timeframe = [file for file in files if file.endswith('sequences.csv')]
    assert len(timeframe) == 1, f'There is {"more" if len(timeframe) > 1 else "less"} than one sequence file.'
    timeframe = pd.read_csv(timeframe[0], index_col=0, header=0, parse_dates=[1, 2])
    timeframe = timeframe.sort_index()

    # sort the files into normal ones and the sequence information
    sequences = [pd.DataFrame() for _ in range(len(sequencefiles))]
    with mp.Pool(mp.cpu_count()//2) as pool:
        desc = "Loading files into memory"
        for index, df in tqdm(pool.imap_unordered(read_file, sequencefiles), desc=desc, total=len(sequencefiles)):
            sequences[index] = df
    assert all(ele.shape[0] != 0 for ele in sequences), 'Some data was not loaded correctly.'
    sequence_amount = len(sequencefiles)

    # go through the sequences and find the most common number of values
    sequence_length = int(np.median(np.array([sequence.shape[0] for sequence in sequences])))

    # go through all the dataframes and delete unfilled columns as well as resampling them to a given length
    processer = functools.partial(process_df, target_length=sequence_length)
    with mp.Pool(mp.cpu_count() // 2) as pool:
        desc = "Processing and resampling sequences"
        for index, df in tqdm(enumerate(pool.imap(processer, sequences)), desc=desc, total=len(sequences)):
            sequences[index] = df

    # go through the sequences and put them into a dict which then gets fused into matrices for each array
    sequence_dict = collections.defaultdict(dict)
    for idx, sequence in enumerate(sequences):
        for name, col in sequence.items():
            sequence_dict[name][idx] = col.to_numpy()
    del sequences

    # once we have the sequences, we can construct the matrices
    sequence_indices = dict()
    sequence_matrices = {name: np.empty((sequence_length, len(vals))) for name, vals in sequence_dict.items()}
    for name, vals in sequence_dict.items():

        # sort the indices
        indices = sorted(list(vals.keys()))
        sequence_indices[name] = indices

        # put the values into the matrices
        assert sequence_matrices[name].shape[1] == len(indices), 'Something is off.'
        for idx, ind in enumerate(indices):
            sequence_matrices[name][:, idx] = vals[ind]

    # go over the sequence matrices and compute the values we need
    eigensequences = dict()
    desc = 'Computing decompositions'
    delete_signals = []
    for name, matrix in tqdm(sequence_matrices.items(), desc=desc):
        if matrix.shape[1] < 20:
            delete_signals.append(name)
            continue
        result = fbpca.pca(matrix, k=number_eigenvectors, raw=True, n_iter=3, l=number_eigenvectors * 3)
        eigensequences[name] = result

    # delete some of the unnecessary indices as there were not enough sequences
    for name in delete_signals:
        del sequence_indices[name]

    # fill array of participation for all the eigensequences participations
    return sequence_matrices, sequence_indices, eigensequences, timeframe


@functools.cache
def load_data_cached(number_eigenvectors: int = 5):
    return load_data(number_eigenvectors)


@functools.cache
def construct_heatmap(eig_index1: int = 0, eig_index2: int = 1, eig_index3: int = 2, diff_view=False) -> (dict[str:int], np.ndarray):

    # get the data
    sequence_matrices, sequence_indices, eigensequences, timeframe = load_data_cached()

    # make the indices
    required_eigensequene_indices = (eig_index1, eig_index2, eig_index3)

    # check whether we computed enough eigenvalues for the specified indicies
    necessary_eigensequences = max(required_eigensequene_indices)
    assert all(result[0].shape[1] >= necessary_eigensequences for result in eigensequences.values()), \
        'We have not computed enough eigensequences for the signals.'

    # find the largest sequence number that we still got left over
    max_idx = max(ele for indices in sequence_indices.values() for ele in indices)

    # construct the matrix that will be plotted in the heatmap
    values = np.empty((len(sequence_indices.keys()), max_idx + 1, 3))
    values.fill(np.NaN)
    name_indices = {name: idx for idx, name in enumerate(sequence_indices.keys())}
    for name, eigensequence in eigensequences.items():
        name_index = name_indices[name]
        idces = sequence_indices[name]
        for vector_index in required_eigensequene_indices:
            participation = normalization(eigensequence[2][vector_index, :])
            values[name_index, idces, vector_index] = participation

    if diff_view:
        values = np.abs(np.diff(values, axis=1))
        values = np.concatenate((values[:, 0:1, :], values), axis=1)
    return name_indices, values, timeframe


def compute_mean_sequence(sequence_matrix: np.ndarray, sequence_indices: list, start_idx: int, end_idx: int) \
        -> np.ndarray:

    # find the index itself or one larger
    idx_left = bisect.bisect_left(sequence_indices, start_idx)
    idx_right = bisect.bisect_left(sequence_indices, end_idx)

    # check whether there is data available
    if idx_left > idx_right:
        result = None
    elif idx_left == idx_right:
        result = sequence_matrix[:, idx_left]
    else:
        result = np.mean(sequence_matrix[:, idx_left:idx_right+1], axis=1)
    return result


def draw_heatmap(keep_columns: list[str] = None, diff_view: bool = False):

    # get the data for the heatmap
    name_indices, values, timeframe = construct_heatmap(diff_view=diff_view)
    if keep_columns is None:
        keep_columns = list(name_indices.keys())

    # alphabetically sort the columns
    keep_columns.sort()

    # construct the new array
    new_indices = list(map(name_indices.get, keep_columns))
    values = values[new_indices, :]

    #  update the trace for a hover template
    # https://stackoverflow.com/questions/59057881/how-to-customize-hover-template-on-with-what-information-to-show
    # https://community.plotly.com/t/how-to-customize-the-hovering-text-when-it-is-another-column-rather-than-the-columns-introduced-for-axes/70717
    # https://stackoverflow.com/questions/73649907/plotly-express-imshow-hover-text/73658192#73658192
    # https://chart-studio.plotly.com/~empet/15366/customdata-for-a-few-plotly-chart-types/#/

    # make the plot and create the customdata
    # fig = px.imshow(values, aspect='auto')
    signal_array = np.repeat(np.array(keep_columns)[:, None], axis=1, repeats=values.shape[1])
    time_array = np.repeat(np.array([f"From {timeframe.loc[idx, 'from']} to {timeframe.loc[idx, 'to']}"
                                     for idx in range(timeframe.index.min(), timeframe.index.max()+1)])[None, :],
                           axis=0,
                           repeats=values.shape[0])
    customdata = np.dstack((signal_array, time_array, values))

    # convert the values to RGB and take care of NaN
    tmp_z = np.rint(values*255)
    tmp_z = tmp_z.astype(np.int32)

    # make the figure
    fig = go.Figure(go.Image(z=tmp_z, customdata=customdata,
                             hovertemplate="<br>".join([
                                 "Sequenz: %{x}",
                                 "Signal: %{customdata[0]}",
                                 "%{customdata[1]}",
                                 "Participation: %{customdata[2]:.2%}, %{customdata[3]:.2%}, %{customdata[4]:.2%}"
                             ])))

    # Define dragmode, newshape parameters, amd add modebar buttons
    fig.update_layout(
        # dragmode='drawrect',  # define dragmode
        dragmode='zoom',
        newshape=dict(line_color='cyan'),
        modebar_add=[
                     'drawrect',
                     'eraseshape'
                     ],
        clickmode='event'
    )
    fig.update_xaxes(scaleanchor=False)
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(keep_columns))),
            ticktext=keep_columns
        )
    )

    # picture size
    # https://community.plotly.com/t/resizing-issue-for-px-imshow-figure/70266/2
    fig.update_layout(margin={"t": 0, "b": 0, "r": 0, "l": 0, "pad": 0})

    # Add modebar buttons
    return fig


# dash app layout ------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # load the data into the cache
    load_data_cached()

    external_stylesheets = [dbc.themes.BOOTSTRAP]
    app = Dash(__name__, external_stylesheets=external_stylesheets,
               external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/dragula/3.7.2/dragula.min.js"],
               )

    # how to call an update from button presses modified from
    # https://community.plotly.com/t/how-to-get-trigger-from-keyboard/76960/6
    app.clientside_callback(
        ClientsideFunction(namespace="clientside", function_name="delete_stuff"),
        Output("signal-graph", "id"),
        Input("signal-graph", "id")
    )

    # how to make draggable signals
    # https://community.plotly.com/t/ordering-children-with-drag-drop-how-to-trigger-dash-component-has-changed-from-js/47243/10
    app.clientside_callback(
        ClientsideFunction(namespace="clientside", function_name="make_draggable"),
        Output("drag-container", "data-drag"),
        [Input("drag-container", "id"),
         Input("drag-ignore-container", "id")],
    )

    # textbox selection with regex
    app.clientside_callback(
        ClientsideFunction(namespace="clientside", function_name="select_signals"),
        Output("signal-include-container", "children"),
        [Input("signal-include-text-search", "value"),
         Input("drag-container", "id"),
         Input("drag-ignore-container", "id"),
         Input("drag-container", "id")
         ]
    )
    app.clientside_callback(
        ClientsideFunction(namespace="clientside", function_name="select_signals"),
        Output("signal-ignore-container", "children"),
        [Input("signal-ignore-text-search", "value"),
         Input("drag-ignore-container", "id"),
         Input("drag-container", "id"),
         Input("drag-container", "id")
         ],
    )

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

    # define an event for our custom event listener to drag and drop buttons
    drop_event = {"event": "dropcomplete", "props": ["detail.name", "detail.children"]}
    delete_event = {"event": 'shapeDeletion', "props": ["detail.children"]}

    # get the explanation text from the file
    with open("assets/explanation_sequences.txt") as filet:
        text = filet.read()

    # define the app layout
    app.layout = html.Div([
        html.Div(children=[
            html.H1(f'Characteristic Sequence Analyzer',
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
                                        html.Div(children=[html.Div(col, id=f"signal-{col}", disable_n_clicks=True,
                                                                    className='a')
                                                           for col in load_data_cached()[1].keys()],
                                                 id='drag-container'),
                                        ]),
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
                    dcc.Checklist(
                        ['Difference View'],
                        [],
                        id='checkbox-diff-view',
                        inline=True
                    )
                ],
                    title="Sort and Select columns by dragging and dropping their column names.",
                ),
            ],
                start_collapsed=True,
            )
        ],
            style=styles['div'],
            id='column-container'
        ),
        html.Div(children=[
            "Shortcuts: [DEL] deletes the last drawn rectangle. [0] sets view to anomaly_heatmap. "
            "Any other number scrolls to the regarding selection, e.g. [1].",
            EventListener(children=[
                html.Div(children=[
                    dcc.Graph(
                        id='signal-graph',
                        figure=draw_heatmap(),
                        style={'height': '100%'}
                    ),
                ]),
                dcc.Checklist(
                    ['One Graph'],
                    ['One Graph'],
                    id='checkbox-one-graph',
                    inline=True
                ),
            ],
                events=[delete_event], logging=True, id="delete-listener",
            ),
            html.Button("Delete All", id="delete-button-all", className="button-4"),
        ],
            style=styles['stickydiv'] | styles['div']
        ),
        html.Div(children=[
            ],
                id="overall-div"
            ),
        html.Div(
            children=[],
            style=styles['div'],
            id='text-container',
            hidden=True,
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

    # app callbacks ----------------------------------------------------------------------------------------------------

    @app.callback(
        Output(component_id='text-container', component_property='children', allow_duplicate=True),
        Output(component_id='signal-graph', component_property='figure'),
        Output(component_id='overall-div', component_property='children'),
        Output(component_id='delete-button-all', component_property='children'),
        Output("el", "event"),
        Output("delete-listener", "event"),
        Output('signal-graph', 'clickData'),
        Input(component_id='delete-button-all', component_property='n_clicks'),
        State('signal-graph', 'figure'),
        Input('signal-graph', 'relayoutData'),
        Input(component_id='overall-div', component_property='children'),
        Input("el", "n_events"),
        State("el", "event"),
        State(component_id="drag-container", component_property="children"),
        Input("delete-listener", "n_events"),
        State("delete-listener", "event"),
        Input('signal-graph', 'clickData'),
        Input('checkbox-one-graph', 'value'),
        Input('checkbox-diff-view', 'value'),
        prevent_initial_call=True
    )
    def delete_shapes(delete_all, current_figure, relayout_data, all_children,
                      nevents, event_data, children,
                      n_delete_events, delete_event_data, click_data, checkbox, checkbox_diff):

        # get the original data
        sequence_matrices, sequence_indices, eigensequences, timeframe = load_data_cached()

        # get the current colum names
        columns = current_figure['layout']['yaxis']['ticktext']

        # get the new columns
        if event_data is not None:
            new_cols = [ele.split('-')[1] for ele in event_data['detail.children']]
        else:
            new_cols = columns

        # get existing shapes and sort out all that are not rect
        if len(columns) == len(new_cols):
            shapes = [ele for ele in current_figure['layout'].get('shapes', []) if ele['type'] == 'rect']
        else:
            shapes = []

        # create a shape if we have clicked one
        if click_data is not None:
            point = [click_data['points'][0]['x'], click_data['points'][0]['y']]
            new_shape = {'editable': True, 'visible': True, 'showlegend': False, 'legend': 'legend', 'legendgroup': '',
                         'legendgrouptitle': {'text': ''}, 'legendrank': 1000,
                         'label': {'text': f'Selection {len(shapes)}', 'texttemplate': '', 'font': {'color': 'white'}},
                         'xref': 'x', 'yref': 'y', 'layer': 'above', 'opacity': 1,
                         'line': {'color': 'cyan', 'width': 4, 'dash': 'solid'}, 'fillcolor': 'rgba(0, 0, 0, 0)',
                         'fillrule': 'evenodd', 'type': 'rect', 'x0': point[0], 'y0': point[1],
                         'x1': point[0], 'y1': point[1]}
            shapes.append(new_shape)

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
            shapes.clear()

        # update the shape accordingly (by snapping to complete integers) and only keep shapes that we like
        for idx, shape in enumerate(shapes):

            # set the indices
            shape['y0'] = round(shape['y0'])
            shape['y1'] = round(shape['y1'])

            # give the shape a text
            shape['label']['text'] = f'Selection {idx + 1}'
            shape['label']['font'] = {'color': 'white'}

        # check whether we only want one graph (and then make that graph)
        one_graph = len(checkbox) >= 1
        df = dict()

        # make the other plots for each of the shapes
        all_children.clear()
        for idx, shape in enumerate(shapes):

            # get all the columns we need to select (depending on where we start, one of the shapes is the larger one)
            column_start = shape['y0']
            column_end = shape['y1']
            column_start, column_end = sorted((column_start, column_end))
            column_start = max(column_start, 0)
            column_end = min(column_end, len(columns)-1)

            # get the time we need to select (depending on where we start, one of the shapes is the larger one)
            start_idx = shape['x0']
            end_idx = shape['x1']
            start_idx, end_idx = sorted((start_idx, end_idx))

            # make a dict to save the signals (reset it if only want one graph)
            if not one_graph:
                df = dict()

            # compute all the signals with different names and save it into a dataframe
            for name in columns[column_start:column_end+1]:
                result = compute_mean_sequence(sequence_matrices[name], sequence_indices[name], start_idx, end_idx)

                # check whether we only want one figure, then we need descriptive names
                if one_graph:
                    name = f'Selection {idx+1}: {name}'

                # save the computed signal into the dict
                if result is not None:
                    df[name] = result

            # make the plot for each signal if we want multiple graphs
            if not one_graph or idx == len(shapes)-1:

                # create the dataframe from the dict
                df = pd.DataFrame(df)

                # make the plots!
                fig = px.line(df, x=df.index, y=df.columns)
                fig.update_layout(xaxis_title='Time', yaxis_title='Signal Value (MinMax normalized)')

                # make the title of the div we are about to create
                if not one_graph:
                    sequence_title = f"Sequence {int(start_idx)} to {int(end_idx)}" if start_idx != end_idx \
                        else f"Sequence {start_idx}"
                    complete_title = f"Signals for Selection {idx+1}: {sequence_title}"
                    time_frame = (f'From {timeframe.loc[int(start_idx), "from"]} to '
                                  f'{timeframe.loc[int(start_idx), "to"]}')
                else:
                    complete_title = f"Signals for all Selections"
                    time_frame = ""

                # create a new div that contains the plot
                all_children.append(html.Div(children=[html.H3(complete_title),
                                                       time_frame,
                                                       html.Details(children=[
                                                                        html.Div(children=[
                                                                            dcc.Graph(figure=fig)])
                                                                    ],
                                                                    open=True,
                                                                    id=f'signal-selection{idx + 1}'
                                                                    )],
                                             style=styles['div']
                                             )
                                    )

        # update figure layout (shapes and also the ordering on the yaxis)
        current_figure = draw_heatmap(new_cols, diff_view=bool(checkbox_diff))
        current_figure.update_layout(shapes=shapes)
        if 'xaxis.range[0]' in relayout_data and 'yaxis.range[0]' in relayout_data:
            current_figure.update_layout(xaxis_range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']],
                                         yaxis_range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']])

        # create the string when we clicked
        click_string = f"Klicked at {pd.Timestamp('today')}"

        return click_string, current_figure, all_children, f"Delete all {len(shapes)}", None, None, None

    app.run(debug=False, host='0.0.0.0', port=8081)
