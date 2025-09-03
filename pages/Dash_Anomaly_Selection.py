import logging
import time
import sys

from dash import dcc, html, Input, Output, State, ctx, ClientsideFunction, register_page, clientside_callback, callback
from dash.dash_table.Format import Format, Group, Scheme, Symbol
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import util.load_data as utl
from GLOBALS import *
import util.cache_registry as ucache
import util.residuals as procd


# register the page to our application
register_page(__name__, path="/anomaly")
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


def table_type(df_column):
    # Note - this only works with Pandas >= 1.0.0

    if sys.version_info < (3, 0):  # Pandas 1.0.0 does not support Python 2
        return 'any'

    if pd.api.types.is_datetime64_any_dtype(df_column.dtype):
        return 'datetime',
    elif pd.api.types.is_string_dtype(df_column.dtype):
        return 'text'
    elif pd.api.types.is_numeric_dtype(df_column.dtype):
        return 'numeric'
    else:
        return 'any'


@ucache.lru_cache(maxsize=5)
def get_anomaly_data(session_id: str, folder_name: str, scoring_fn, k_neighbors: int = 3, correlation_threshold: float = 0.85) -> (int, float, pd.DataFrame, dict[str: pd.DataFrame], dict[str: pd.DataFrame]):
    start = time.perf_counter()

    # get the data from the disk
    _, regression_results_grouped, max_correlation = utl.preprocess_regression_results(session_id, folder_name)
    scores, _, _, _, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # build the dataframe we want to create
    data_collection = {'Anomaly Score': [],
                       'Signal Name': [],
                       'Block': [],
                       'Turbine': [],
                       'Component': [],
                       'Measurement': [],
                       'Neighbors': [],
                       'Neighbor Names': [],
                       'Largest Correlation': [],
                       'Smallest Correlation': []
                       }
    top_n_grouped = dict()
    residual_scores = dict()
    for idx, (name, anomaly_score) in enumerate(regression_results_grouped, 1):

        # check for the largest correlation to any other neighbor
        if max_correlation.loc[name, "correlation"] < correlation_threshold:
            continue

        # get the largest neighbors
        regression_results = regression_results_grouped.get_group(name)
        n_largest_neighbors = regression_results.nlargest(k_neighbors, "correlation")
        n_largest_neighbors = n_largest_neighbors[n_largest_neighbors['correlation'] >= correlation_threshold]
        top_n_grouped[name] = n_largest_neighbors

        # get the neighbor names
        neighbors = n_largest_neighbors["y"]
        neighbors_list = neighbors.to_list()
        neighbors_list.append(name)

        # compute the residuals
        result_df = procd.compute_weighted_residual_norm(regression_results, neighbors_list, scores, target=name, coming_from=f'anomaly-selection-{k_neighbors=}-{correlation_threshold=}-{scoring_fn=}-{name=}')
        residual_scores[name] = result_df[name]

        # compute the anomaly score from the residuals
        anomaly_score = procd.get_registered_scoring_functions()[scoring_fn](result_df[name])

        # append the name and the corresponding anomaly score of the signal
        data_collection['Anomaly Score'].append(anomaly_score)
        data_collection['Signal Name'].append(name)

        # collect the information from the KKS name
        data_collection['Block'].append(name[1:2])
        data_collection['Turbine'].append(name[2:3])
        data_collection['Component'].append(name[3:8])
        data_collection['Measurement'].append(name[8:10])

        # go through all neighbors and keep the correlation information
        neighbors_vals = [f'{row["y"]}: {row["correlation"]:0.3f}' for _, row in n_largest_neighbors.iterrows()]
        neighbors_names = [row["y"] for _, row in n_largest_neighbors.iterrows()]
        data_collection['Neighbors'].append(", ".join(neighbors_vals))
        data_collection['Neighbor Names'].append(", ".join(neighbors_names))
        data_collection['Largest Correlation'].append(n_largest_neighbors['correlation'].max())
        data_collection['Smallest Correlation'].append(n_largest_neighbors['correlation'].min())
    data_collection = pd.DataFrame(data_collection)
    data_collection.sort_values("Anomaly Score", ascending=False, inplace=True)

    logger.info(f"[{__name__}] Creating the anomaly selection dataframe took {time.perf_counter() - start:0.2f} s.")
    return k_neighbors, correlation_threshold, data_collection, top_n_grouped, residual_scores


def layout(session_id: str = "", folder_name: str="", **kwargs):

    # check whether we have a folder
    if not folder_name:
        return html.H1("Please upload a file using the sidebar.")

    # check whether we have all necessary data
    _, _, _, anomaly_scores, distances, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))
    if anomaly_scores is None or distances is None:
        return html.H1(
            "Your zip-file does not contain the anomaly_scores.parquet or/and the distances.csv file(s).")

    # get the scoring function
    scoring_fns = procd.get_registered_scoring_functions()
    standard_scoring_fn_name, _ = procd.get_standard_scoring_functions()

    # get the preprocessed data
    k_neighbors, correlation_threshold, data_collection, _, _ = get_anomaly_data(session_id, folder_name, standard_scoring_fn_name)
    _, regression_results_grouped, _ = utl.preprocess_regression_results(session_id, folder_name)
    k_neighbors_list = list(range(1, regression_results_grouped.size().max()+1))

    # get the explanation text from the file
    text = "Do not know what to write here yet."

    # define the app layout
    layout_definition = (
        html.Div([
            html.Div(children=[
                html.H1(f'Anomaly Selection',
                        style={'fontSize': 40}
                        ),
                dbc.Accordion(children=[
                    dbc.AccordionItem(children=[html.Pre(text, className='pre-expl')], title="Explanation", )
                ],
                    start_collapsed=True,
                ),
                "How many neighors do you want to check?",
                dbc.Select(id="select-neighbor-size", options=sorted(k_neighbors_list), value=min(k_neighbors_list[-1], k_neighbors), persistence="session"),
                html.Div(children=f'Correlation Value: {correlation_threshold}', id='anomaly-selection-corr-val-div'),
                dcc.Slider(min=0, max=0.99,
                           value=correlation_threshold,
                           id='anomaly-selection-correlation-slider',
                           persistence="session"),
                "Choose scoring function",
                dbc.Select(id="select-scoring-fn", options=list(scoring_fns.keys()), value=standard_scoring_fn_name, persistence="session"),
            ],
                style=dash_heatmap_styles['div'],
            ),
            html.Div(children=[
                dbc.Label(['For folder operations see:  ', html.A("here", href='https://dash.plotly.com/datatable/filtering', target="_blank")]),
                dcc.Loading(
                    id="anomaly-selection-loading-signals",
                    children=[
                        dash.dash_table.DataTable(data=data_collection.to_dict('records'),
                                                  columns=[{"name": i,
                                                            "id": i,
                                                            'type': table_type(data_collection[i]),
                                                            'format': Format(
                                                                precision=4,
                                                                scheme=Scheme.fixed,
                                                                group=Group.yes,
                                                                groups=3,
                                                                group_delimiter=',',
                                                                decimal_delimiter='.',)
                                                            if table_type(data_collection[i]) == 'numeric' else None,
                                                            }
                                                           for i in data_collection.columns],
                                                  id='anomaly-selection-tbl',
                                                  filter_action="native",
                                                  sort_action="native",
                                                  sort_mode="multi",
                                                  column_selectable="single",
                                                  row_selectable="single",
                                                  selected_columns=[],
                                                  selected_rows=[],
                                                  page_action="native",
                                                  page_current=0,
                                                  page_size=10,
                                                  )
                    ]
                ),
            ],
                style=dash_heatmap_styles['div'],
                id='anomaly-selection-tbl-container'
            ),
            html.Div(children=[
                "Shortcuts: [DEL] deletes the last drawn rectangle. [0] sets view to anomaly_heatmap. Any other number scrolls to "
                "the regarding selection, e.g. [1].",
                dcc.Graph(
                    id='anomaly-selection-residual-graph',
                    figure=None,
                    style={},
                ),
            ],
                style=dash_heatmap_styles['stickydiv'] | dash_heatmap_styles['div'],
                id='anomaly-selection-residual-div',
                hidden=True,
            ),
            html.Div(
                children=[
                    dcc.Graph(
                        id='anomaly-selection-signal-graph',
                        figure=None,
                        style={},
                    )
                ],
                style=dash_heatmap_styles['div'],
                id='anomaly-selection-signal-div',
                hidden=True,
            ),
            html.Div(
                children=[],
                style=dash_heatmap_styles['div'],
                id='anomaly-selection-signal-include-container',
                hidden=True,
            ),
            html.Div(
                children=[],
                style=dash_heatmap_styles['div'],
                id='anomaly-selection-signal-ignore-container',
                hidden=True,
            ),
        ],
        ))
    return layout_definition


@callback(
    Output('anomaly-selection-corr-val-div', 'children'),
    Output('anomaly-selection-tbl', 'data'),
    Input("session-id", "data"),
    Input("folder-name", "data"),
    Input("select-neighbor-size", "value"),
    Input("anomaly-selection-correlation-slider", "value"),
    Input('select-scoring-fn', 'value'),
)
def adapt_anomaly_data(session_id: str, folder_name: str, k_neighbors: str, correlation_threshold: float, scoring_fn_name: str):

    # process the files
    k_neighbors = int(k_neighbors)
    _, _, data_collection, _, _ = get_anomaly_data(session_id, folder_name, scoring_fn_name, k_neighbors, correlation_threshold)
    return f'Correlation Value: {correlation_threshold}', data_collection.to_dict('records')


@callback(
    Output('anomaly-selection-residual-graph', 'figure'),
    Output('anomaly-selection-residual-div', 'hidden'),
    Output("anomaly-selection-signal-graph", 'figure'),
    Output("anomaly-selection-signal-div", 'hidden'),
    Input("session-id", "data"),
    Input("folder-name", "data"),
    Input('anomaly-selection-tbl', "derived_virtual_selected_rows"),
    Input('anomaly-selection-tbl', "derived_virtual_data"),
    State("select-neighbor-size", "value"),
    State("anomaly-selection-correlation-slider", "value"),
    State('select-scoring-fn', 'value'),)
def create_selection(session_id: str, folder_name: str, selected_rows, data, k_neighbors: str, correlation_thresh: float, scoring_fn_name: str):

    # check whether we selected nothing
    if selected_rows is None or not selected_rows:
        return go.Figure(go.Scatter(x=[], y=[])), True, go.Figure(go.Scatter(x=[], y=[])), True
    selection = data[selected_rows.pop()]['Signal Name']

    # transform the neighbors
    k_neighbors = int(k_neighbors)

    # get the anomaly residuals
    _, _, _, top_n_grouped, residual_scores = get_anomaly_data(session_id, folder_name, scoring_fn_name, k_neighbors, correlation_thresh)

    # get the scores and raw signals
    _, _, _, _, _, raw_signals_grouped, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # get the neighbor names
    neighbors = top_n_grouped[selection]["y"]
    neighbors_list = neighbors.to_list()
    neighbors_list.append(selection)

    # load the raw signals
    raw_signals = pd.concat(raw_signals_grouped.get_group(grp) for grp in neighbors_list)

    # make a colormap for both figures
    # https://stackoverflow.com/a/68209157
    c = dict(zip(neighbors_list, px.colors.qualitative.G10))

    # make the figure from the signals
    fig = px.line(residual_scores[selection], color_discrete_map=c)
    fig2 = px.line(raw_signals, y="normalized value", color='sensor', color_discrete_map=c, hover_data="value")

    # reorder the lines so the sensor itself is plotted last and the neighbors are in order
    sort_keys = {name: idx for idx, name in enumerate(neighbors_list)}
    fig.data = sorted(fig.data, key=lambda x: sort_keys[x['name']])
    fig2.data = sorted(fig2.data, key=lambda x: sort_keys[x['name']])

    # customize the legend and labels
    fig.update_layout(legend_title='residuals')
    fig.update_layout(xaxis_title="Date", yaxis_title="Residual Value (Unexpected Change)")
    fig2.update_layout(xaxis_title="Date", yaxis_title="Normalized Signal Values")
    return fig, False, fig2, False


@callback(
    Output(component_id="anomaly-selection-residual-graph", component_property="figure", allow_duplicate=True),
    Output(component_id="anomaly-selection-signal-graph", component_property="figure", allow_duplicate=True),
    Input(component_id="anomaly-selection-residual-graph", component_property="clickData"),
    State(component_id="anomaly-selection-residual-graph", component_property="figure"),
    State(component_id="anomaly-selection-signal-graph", component_property="figure"),
    prevent_initial_call=True)
def click_in_residuals(click_data, residual_fig, signal_fig):

    # check whether we selected nothing
    if click_data is None:
        return residual_fig, signal_fig

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