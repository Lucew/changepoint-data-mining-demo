import plotly.express as px
import util.create_tsne as utsne


def prepare_plot_data(session_id: str, folder_name: str, perplexity: int = None, correlation_threshold: int = None, window_size: int = None,
                      selected_components: list[str] = None, selected_measurements: list[str] = None, selected_signals: list[str] = None):
    # load the data
    bokeh_df, _, _ = utsne.create_tsne(session_id, folder_name, perplexity, correlation_threshold, window_size)

    # check whether we need to select something
    if selected_components is not None:

        # select the data we want to have
        selected_components = set(selected_components)
        bokeh_df = bokeh_df[bokeh_df['component'].isin(selected_components)]
    if selected_measurements is not None:

        # select the data we want to have
        selected_measurements = set(selected_measurements)
        bokeh_df = bokeh_df[bokeh_df['measurement'].isin(selected_measurements)]

    if selected_signals is not None:

        # select the data we want to have
        selected_signals = set(selected_signals)
        bokeh_df = bokeh_df[bokeh_df['original'].isin(selected_signals)]
    # print("left", bokeh_df[["original", "x", "y"]])
    return bokeh_df

def create_scatter(session_id: str, folder_name: str, perplexity: int = None, correlation_threshold: float = None, window_size: int = None,
                   selected_components: list[str] = None, selected_measurements: list[str] = None, selected_signals: list[str] = None):

    # get the scatter plot information
    bokeh_df = prepare_plot_data(session_id, folder_name, perplexity, correlation_threshold, window_size, selected_components, selected_measurements, selected_signals)

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
                      perplexity: int = None, correlation_threshold: float = None, window_size: int = None,
                      selected_components: list[str] = None, selected_measurements: list[str] = None):

    # get the scatter plot information
    bokeh_df = prepare_plot_data(session_id, folder_name, perplexity, correlation_threshold, window_size, selected_components, selected_measurements)

    # make the figure
    # NOTE: Linking the 3D plot to the 2D is not trivial as selectedpoints is not a valid option
    # for scatter_3d as of now
    fig = px.scatter_3d(bokeh_df, x="x3d", y="y3d", z='z3d', color="component", symbol="measurement",
                        hover_data=["original"])

    return fig