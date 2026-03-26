import dash
from dash import html

dash.register_page(__name__, path='/')


def layout(session_id: str, folder_name: str, **kwargs):
    # check whether we have a folder
    if not folder_name:
        return html.H1("Please upload a file using the sidebar.")
    return html.Div(html.H1(["Navigate to one of the Pages in the Sidebar"]), style={"display": "flex", "justifyContent": "center", "alignItems": "center",  "minHeight": "100vh", "margin": "0"})