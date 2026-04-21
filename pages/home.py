import dash
from dash import html

import util.styles as ustyle


dash.register_page(__name__, path='/')


def layout(session_id: str, folder_name: str, **kwargs):
    # check whether we have a folder
    idea_text = "Navigate to one of the Pages in the Sidebar"
    if not folder_name:
        idea_text = "Please upload a file using the sidebar"

    # make a style update
    style_update = {"display": "flex", "justifyContent": "center", "alignItems": "center", "height": "100vh"}
    return html.Div([
        html.Div([
            html.H1("Welcome to Changepoint!"),
            html.Br(),
            html.Div(
                idea_text,
                style={
                    "display": "flex", "justifyContent": "center", "alignItems": "center", "fontSize": "Large"
                }),
            html.Br(),
            html.Div(
                children=[
                    "For further details, watch our introductory ",
                    html.A("video", href="https://www.fau.tv/clip/id/63306", target="_blank"),
                    " or visit the source ",
                    html.A("code.", href="https://github.com/Lucew/changepoint-data-mining-demo", target="_blank"),
                ],
                style={
                    "display": "flex", "justifyContent": "center", "alignItems": "center", "fontSize": "small", "whiteSpace": "pre",
                }),
            html.Br(),
            html.Br()
        ],
            style=ustyle.div_styles['div'] | {"marginTop": "auto", "width": "30%"})] ,
        style=style_update)