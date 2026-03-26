import pandas as pd
import plotly.express as px

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