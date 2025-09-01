import os.path
import typing
import zipfile
import io
import base64
import uuid
import shutil
import logging

import dash
import plotly
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd

import util.load_data as uld
from GLOBALS import *


# Page descriptions
page_info = {
    "Heatmap": "This page allows to analyze CP scores of multiple signals as heatmap.",
    "Scatter": "This page allows to analyze CP correlation in scatter plots with anomaly scores.",
    "Correlate": "This page gives you export options and reporting tools."
}

# get the logger
# https://stackoverflow.com/q/3220284
logger = logging.getLogger("frontend-logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)
ch.setLevel(logging.DEBUG)

# log some versions
logger.info(f"Dash version: {dash.__version__}.")
logger.info(f"Plotly version: {plotly.__version__}.")

# Dash app setup
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]
app = dash.Dash(__name__, use_pages=PAGED, external_stylesheets=external_stylesheets,
                external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/dragula/3.7.2/dragula.min.js"],
                routing_callback_inputs={
                    "session_id": Input('session-id', "data"),
                    "folder_name": Input('folder-name', "data"),
                },
                suppress_callback_exceptions=True,
                )
app.title = "Changepoint"

# App layout
side_bar_content = dbc.Container([
    html.Div(html.H1("Changepoint", className="my-4 text-center")),

    # give the session an id and save the files
    dcc.Store(data=__name__ if APPLICATION_LEVEL == Level.DEBUG else str(uuid.uuid4()), id='session-id', storage_type='session'),
    dcc.Store(data="", id='file-name', storage_type='session'),
    dcc.Store(data="", id='folder-name', storage_type='session'),
    dcc.Store(data="", id='upload-status', storage_type='session'),

    # Upload component
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.P(children=[DEFAULT_UPLOAD_TEXT], id='upload-text', style={"fontSize": "20px", "marginBottom": "5px", "overflow": "flex"}),
                html.Small("Drag and drop or click to browse", style={"color": "#6c757d"})
            ]),
            className="upload-box",
            multiple=False,
        ),
    ], style={"marginBottom": "15px"}, className="upload-box-div"),

    # Display uploaded filename
    dbc.Button("Delete File.", id="delete-file-button", disabled=True, style={"margin": "1px"}),
    dbc.Button("Delete ALL Files.", id="delete-all-file-button", style={"margin": "1px"}),

    # Responsive grid of page buttons
    html.Div(
        id="page-button-grid",
        children=[
            dbc.Button(title, href=f"/{title.lower()}", style={"margin": "1px"})
            for title, desc in page_info.items()
        ],
    ),

    # Collapsible container (hidden initially)
    dbc.Accordion(
        children=[
            dbc.AccordionItem(title=f"{title} Description", children=desc, style={"marginTop": "10px"})
            for i, (title, desc) in enumerate(page_info.items(), start=1)
        ],
        style={"display": "none"},
        start_collapsed=True,
        id="page-link-container",
    ),
], style={"maxWidth": "25rem"}, fluid=True)

# set the final layout
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(children=[html.Div(html.H1(id="sidebar-symbol")),
                       html.Div(side_bar_content, className='sidebar-content-div')],
             className="sidebar",),
    dash.page_container],
    id="overall-page-container",)


# upload the zipfile and unpack it
@app.callback(
    Output("file-name", "data", allow_duplicate=True),
    Output("folder-name", "data", allow_duplicate=True),
    Output("upload-status", "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    Input('session-id', 'data'),
    prevent_initial_call=True,
)
def handle_zip_upload(contents: str, filename: str, session_id: str):
    # check the obvious problems
    if not filename:
        return "", "", ""
    if not filename.lower().endswith(".zip"):
        return "", "", "Wrong filetype"

    try:
        # Decode uploaded content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # Read zip file from memory
        zip_buffer = io.BytesIO(decoded)
        with zipfile.ZipFile(zip_buffer, 'r') as zipref:

            # create the path to write to
            output_folder = os.path.join(DATA_FOLDER, f"{session_id}")

            # remove the old files
            if os.path.isdir(output_folder):
                shutil.rmtree(output_folder)

            # go through the files and make some sanity checks
            namelist = zipref.namelist()
            folders = set(os.path.split(name)[:-1] for name in namelist)

            # check that there is only one folder
            assert len(folders) == 1, f'There should be only one folder in your ZIP file. You have: "{folders}".'
            folder = folders.pop()[0]

            # write the data to the folder
            for member in namelist:
                zipref.extract(member=member, path=output_folder)

        # clear function cache and load the data to check for any errors
        uld.load_data.cache_clear()
        load_files(session_id, folder)

    except Exception as e:
        logger.error(e)
        return "", "", "Error loading files"

    # log the success
    logger.info(f"Uploaded file {filename} with folder {folder} from session {session_id}.")

    # create the success returns
    return filename, folder, ""


@app.callback(
    Output("folder-name", "data", allow_duplicate=True),
    Output("file-name", "data", allow_duplicate=True),
    Output("upload-status", "data", allow_duplicate=True),
    Output("upload-data", "contents"),
    Output("upload-data", "filename"),
    Input("delete-file-button", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_button_click(n: int, session_id: str):

    # nothing is clicked
    if n is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # delete the files
    filename = os.path.join(DATA_FOLDER, session_id)
    shutil.rmtree(filename)

    # clear the function cache
    uld.load_data.cache_clear()

    # log the deletion
    logger.info(f"Deleted files from session {session_id}.")

    # reset the folder
    return "", "",  "", None, None



# a function to check available files
@app.callback(
    Output("delete-file-button", "disabled"),
    Output("page-link-container", "style"),
    Output("page-button-grid", "style"),
    Output("upload-text", "children"),
    Output("sidebar-symbol", "children"),
    Input("session-id", "data"),
    Input("folder-name", "data"),
    Input("file-name", "data"),
    Input("upload-status", "data"),
)
def available_files(session_id: str, folder_name: str, filename: str, upload_status: str) -> tuple[bool, dict[str:str], dict[str:str], str, str]:

    # check whether we have file
    file_exists = folder_name and os.path.isdir(os.path.join(DATA_FOLDER, session_id, folder_name))

    # load the files into memory
    load_files(session_id, folder_name)

    # create the output
    loaded_file_name = f"Delete File: {filename}" if file_exists else f"Delete File."
    page_link_style = {"display": "block"} if file_exists else {"display": "none"}
    page_button_style = {"display": "flex"} if file_exists else {"display": "none"}

    # check whether we have to make the button active
    button_disabled = not file_exists

    # check the text for the filename
    upload_text = f"✅ Uploaded" if file_exists else DEFAULT_UPLOAD_TEXT

    # the sidebar symbol
    sidebar_symbol = "✔" if file_exists else ("☝" if not upload_status else "❌")

    return button_disabled, page_link_style, page_button_style, upload_text, sidebar_symbol


@app.callback(
    Output("overall-page-container", "children"),
    Input("delete-all-file-button", "n_clicks"),
    State("overall-page-container", "children"),
    prevent_initial_call=True,
)
def initialize_delete_everything(has_clicked, all_page_content: list):
    # https://stackoverflow.com/a/52647370
    # create the password mask
    pwd_mask = html.Div(
        html.Div(
            dbc.Input(id="delete-password-input", placeholder="Deletion Password", type="password"),
            style={"display": "flex", "justify-content": "center", "align-items": "center",
                   "text-align": "center", "width": "30%", "min-height": "100vh"}, ),
        style={"height": "100%", "width": "100%", "position": "fixed !important",
               "z-index": "50000 !important", "left": "0 !important", "top": "0 !important",
               "background-color": "rgba(0,0,0,0.8)", "overflow-x": "hidden", "transition": "0.5s"})
    all_page_content.append(pwd_mask)
    return all_page_content



def load_files(session_id: str, folder_name: str) ->  tuple[typing.Optional[dict[str: pd.DataFrame]], typing.Optional[pd.DataFrame], typing.Optional[tuple[int]], typing.Optional[pd.DataFrame], typing.Optional[pd.DataFrame]]:

    # check whether we have a filename
    logger.info(f"Session {session_id} requested files: [{folder_name if folder_name else "NO FOLDER"}]")
    if not folder_name: return None, None, None, None, None

    # load the files into memory
    scores, signals, window_sizes, anomaly_scores, distances, _, _ = uld.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))
    return scores, signals, window_sizes, anomaly_scores, distances


if __name__ == '__main__':
    logger.info(f"Running Dash Main Page with level {APPLICATION_LEVEL=}.")
    app.run(debug=APPLICATION_LEVEL==Level.DEBUG)
