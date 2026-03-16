import os.path
import typing
import zipfile
import io
import base64
import uuid
import shutil
import logging
import datetime
import argparse

import dash
import plotly
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd

import util.load_data as utl
from GLOBALS import *
import util.cache_registry as ucache


# Page descriptions
page_info = {
    "Heatmap": "This page allows to analyze CP scores of multiple signals as heatmap.",
    "Scatter": "This page allows to analyze CP correlation in scatter plots with anomaly scores.",
    "Correlate": "This page gives you export options and reporting tools.",
    "Anomaly": "This page allows you to scan through the anomalies",
}

# get the logger
# https://stackoverflow.com/q/3220284
logger = logging.getLogger("frontend-logger")
logger.setLevel(logging.DEBUG if APPLICATION_LEVEL == Level.DEBUG else logging.INFO)
ch = logging.StreamHandler()
logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)
ch.setLevel(logging.DEBUG if APPLICATION_LEVEL == Level.DEBUG else logging.INFO)

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

# prepare the folder
def get_first_subfolder(path: str) -> tuple[str, str] | tuple[None, None]:
    """
    Return the first subfolder found inside 'path'.
    If no subfolders exist, return None.
    """
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory.")

    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            return full_path, entry  # return the first folder found
    return None, None

def init():

    # get a session id
    session_id = __name__ if APPLICATION_LEVEL == Level.DEBUG else str(uuid.uuid4())

    # get the first subfolder
    subfolder, relative_path = get_first_subfolder(DATA_FOLDER)
    folder_name = ""
    if subfolder is not None:
        # set the session id to this folder
        session_id = relative_path

        # check for the content of the already existing data
        subsubfolder, relative_path = get_first_subfolder(subfolder)
        if subsubfolder is not None:
            folder_name = relative_path

    file_name = "already-there"
    upload_status = "already-there"
    return session_id, folder_name, file_name, upload_status

# initialize the store variables
__session_id, __folder_name, __file_name, __upload_status = init()

# App layout
side_bar_content = dbc.Container([
    html.Div(html.H1("Changepoint", className="my-4 text-center")),

    # give the session an id and save the files
    dcc.Store(data=__session_id, id='session-id', storage_type='session'),
    dcc.Store(data=__file_name, id='file-name', storage_type='session'),
    dcc.Store(data=__folder_name, id='folder-name', storage_type='session'),
    dcc.Store(data=__upload_status, id='upload-status', storage_type='session'),

    # Upload component
    dcc.Loading(
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
    id='upload-load-spinner', className="upload-load-spinner"),

    # Display uploaded filename
    dbc.Button("Delete File.", id="delete-file-button", disabled=True, style={"margin": "1px"}),
    dbc.Button("Delete ALL Files.", id="delete-all-file-button", style={"margin": "1px"}),
    dbc.Button("Stats.", id="print-cache-stats-button", style={"margin": "1px"}),

    # Responsive grid of page buttons
    html.Div(
        id="page-button-grid",
        children=[
            dbc.Button(title, href=f"/{title.lower()}", style={"margin": "1px"}, id={"type": "disable-btn", "index": title}, disabled=True)
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
app.layout = html.Div(
    [
        dcc.Location(id="url"),

        # store flags + messages
        dcc.Store(id="delete-modal-open", data=False, storage_type="session"),
        dcc.Store(id="delete-notice", data="", storage_type="session"),
        dcc.Store(id="delete-size-bytes", data=0, storage_type="memory"),

        # everything inside here will blur when modal opens
        html.Div(
            id="app-blur-target",
            children=[
                html.Div(
                    children=[
                        dcc.Loading(html.Div(html.H1(id="sidebar-symbol")),),
                        html.Div(side_bar_content, className='sidebar-content-div')
                    ],
                    className="sidebar",
                ),
                dcc.Loading(
                    dash.page_container,
                    className="page-load-spinner",
                    target_components={"_pages_content": "children"},
                ),
            ],
        ),

        # Fullscreen password modal
        # --- Password Modal (compact center) ---
        dbc.Modal(
            id="delete-modal",
            is_open=False,
            backdrop=True,  # click outside won't close it automatically; we close by callbacks
            keyboard=True,  # Esc closes via our 'x' handler if you want to wire it too
            centered=True,
            size="md",  # <— small/medium dialog, not fullscreen
            children=[
                dbc.ModalHeader(
                    children=[
                        dbc.ModalTitle("Confirm full deletion"),
                    ],
                    close_button=True,  # close button
                ),
                dbc.ModalBody(
                    [
                        html.P(
                            "Enter the deletion password to remove ALL files in ./tmp-data-folder.",
                            className="mb-3"
                        ),
                        html.Div(id="delete-size-line", className="text-muted small mb-2"),
                        dcc.Input(
                            id="delete-password-input",
                            type="password",
                            placeholder="Password",
                            autoComplete="off",
                            className="form-control mb-2",  # Bootstrap styling via className
                            n_submit=0,  # so we can capture Enter
                        ),
                        html.Div(
                            id="delete-inline-feedback",
                            className="text-danger small",
                            style={"minHeight": "1.2rem"}
                        ),
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button("Cancel", id="delete-cancel-btn", className="me-2", outline=True),
                        dbc.Button("Delete everything", id="delete-confirm-btn", color="danger"),
                    ]
                ),
            ],
        ),

        # Toast notifications
        dbc.Toast(
            id="delete-toast",
            header="",
            is_open=False,
            dismissable=True,
            duration=4000,
            icon="primary",
            style={"position": "fixed", "top": 20, "right": 20, "zIndex": 1060},
            children=""
        ),
    ],
    id="overall-page-container",
)



# upload the zipfile and unpack it
@app.callback(
    Output("file-name", "data", allow_duplicate=True),
    Output("folder-name", "data", allow_duplicate=True),
    Output("upload-status", "data"),
    Output("url", "pathname", allow_duplicate=True),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    Input('session-id', 'data'),
    prevent_initial_call=True,
)
def handle_zip_upload(contents: str, filename: str, session_id: str):
    # check the obvious problems
    if not filename:
        return "", "", "", "/"
    if not filename.lower().endswith(".zip"):
        return "", "", "Wrong filetype", "/"

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
        ucache.clear_all_caches()
        load_files(session_id, folder)

    except Exception as e:
        logger.error(e)
        return "", "", "Error loading files", "/"

    # log the success
    logger.info(f"Uploaded file {filename} with folder {folder} from session {session_id}.")

    # create the success returns
    return filename, folder, "", "/"


@app.callback(
    Output("folder-name", "data", allow_duplicate=True),
    Output("file-name", "data", allow_duplicate=True),
    Output("upload-status", "data", allow_duplicate=True),
    Output("upload-data", "contents"),
    Output("upload-data", "filename"),
    Output("url", "pathname", allow_duplicate=True),
    Input("delete-file-button", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def delete_own_files(n: int, session_id: str):

    # nothing is clicked
    if n is None:
        return dash.exceptions.PreventUpdate

    # delete the files
    filename = os.path.join(DATA_FOLDER, session_id)
    shutil.rmtree(filename)

    # clear the function cache
    ucache.clear_all_caches()

    # log the deletion
    logger.info(f"Deleted files from session {session_id}.")

    # reset the folder (and redirect to main page)
    return "", "",  "", None, None, "/"



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
    upload_text = DEFAULT_UPLOAD_TEXT_SUCCESS if file_exists else DEFAULT_UPLOAD_TEXT

    # the sidebar symbol
    sidebar_symbol = "✔" if file_exists else ("☝" if not upload_status else "❌")

    return button_disabled, page_link_style, page_button_style, upload_text, sidebar_symbol


@app.callback(
    Output({"type": "disable-btn", "index": dash.ALL}, "disabled"),
    Input("delete-file-button", "disabled"),
    State({"type": "disable-btn", "index": dash.ALL}, "disabled"),
)
def modify_sidebar_buttons(status: bool, button_ids: list):
    return [status]*len(button_ids)


# --- Open modal when "Delete ALL Files." is clicked ---
@app.callback(
    Output("delete-modal", "is_open", allow_duplicate=True),
    Output("delete-modal-open", "data", allow_duplicate=True),
    Output("delete-password-input", "value"),
    Output("delete-inline-feedback", "children"),
    Output("delete-size-bytes", "data"),
    Output("delete-size-line", "children"),
    Input("delete-all-file-button", "n_clicks"),
    prevent_initial_call=True,
)
def open_delete_modal(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    size_bytes = utl.folder_size_bytes(DATA_FOLDER)
    size_text = utl.format_bytes(size_bytes)
    line = f"This will permanently delete approximately {size_text} from {DATA_FOLDER}."

    # open + reset password & inline feedback + show size
    return True, True, "", "", size_bytes, line


# Close on Cancel and the header "x"
@app.callback(
    Output("delete-modal", "is_open", allow_duplicate=True),
    Output("delete-modal-open", "data", allow_duplicate=True),
    Input("delete-cancel-btn", "n_clicks"),
    Input("delete-modal", "is_open"),   # catches “×” and backdrop
    State("delete-modal-open", "data"),
    prevent_initial_call=True,
)
def close_delete_modal(n_cancel, modal_is_open, modal_flag):
    # Cancel clicked
    if n_cancel:
        return False, False

    # Modal just got closed via “×” or backdrop
    if not modal_is_open and modal_flag:
        return False, False

    raise dash.exceptions.PreventUpdate


# --- Blur the entire app while modal is open ---
@app.callback(
    Output("app-blur-target", "style"),
    Input("delete-modal-open", "data"),
)
def blur_background(is_open):
    if is_open:
        # blur + block interactions
        return {
            "filter": "blur(4px)",
            "pointerEvents": "none",
            "transition": "filter 150ms ease-in-out",
        }
    # explicitly restore interactivity (don’t rely on property removal)
    return {
        "filter": "none",
        "pointerEvents": "auto",
        "transition": "filter 150ms ease-in-out",
    }


# --- Confirm deletion: check password, delete, reset stores, toast ---
@app.callback(
    Output("delete-modal", "is_open", allow_duplicate=True),
    Output("delete-modal-open", "data", allow_duplicate=True),
    Output("folder-name", "data", allow_duplicate=True),
    Output("upload-status", "data", allow_duplicate=True),
    Output("delete-toast", "is_open"),
    Output("delete-toast", "header"),
    Output("delete-toast", "icon"),
    Output("delete-toast", "children"),
    Output("delete-inline-feedback", "children", allow_duplicate=True),
    Output("upload-data", "contents", allow_duplicate=True),
    Output("upload-data", "filename", allow_duplicate=True),
    Output("url", "pathname"),
    Input("delete-confirm-btn", "n_clicks"),      # click Confirm
    Input("delete-password-input", "n_submit"),   # press Enter
    State("delete-password-input", "value"),
    State("delete-size-bytes", "data"),
    prevent_initial_call=True,
)
def confirm_delete(n_click_confirm, n_submit, password_value, bytes_planned):
    # if neither happened, do nothing
    if not (n_click_confirm or n_submit):
        raise dash.exceptions.PreventUpdate

    # get the size of the files
    planned_bytes = utl.format_bytes(int(bytes_planned or 0))

    # get the password value
    password_value = password_value or ""
    if password_value != DELETE_PASSWORD:
        # wrong password -> keep modal open, inline error + toast
        return (
            True,  # keep modal open
            True,
            dash.no_update,
            dash.no_update,
            True,              # toast open
            "Deletion failed",
            "danger",
            "Password was incorrect. Nothing was deleted.",
            "Incorrect password. Please try again.",
            dash.no_update,
            dash.no_update,
            dash.no_update,  # no redirect
        )

    removed = utl.delete_all_files_in_root(DATA_FOLDER)

    # logg the progress
    logger.info(f"[Delete-All] Removed {removed} ({planned_bytes}) entries from {DATA_FOLDER} at {datetime.datetime.now(datetime.UTC).isoformat()}Z.")

    # clear the caches
    ucache.clear_all_caches()

    # success: close modal, unblur, reset stores, show toast
    return (
        False,
        False,
        "",     # folder-name
        "",     # upload-status
        True,
        "Deletion complete",
        "success",
        f"Removed {removed} item(s) ({planned_bytes}) from {DATA_FOLDER}.",
        "",     # clear inline error
        None,  # reset upload data field
        None,  # reset upload data field
        "/"
    )


@app.callback(Input("print-cache-stats-button", "n_clicks"),)
def print_cache_stats(n_clicks):
    logger.info(ucache.cache_stats())


def load_files(session_id: str, folder_name: str) ->  tuple[typing.Optional[dict[str: pd.DataFrame]], typing.Optional[pd.DataFrame], typing.Optional[tuple[int]], typing.Optional[pd.DataFrame], typing.Optional[pd.DataFrame]]:

    # check whether we have a filename
    logger.info(f"Session {session_id} requested files: [{folder_name if folder_name else "NO FOLDER"}]")
    if not folder_name: return None, None, None, None, None

    # load the files into memory
    scores, signals, window_sizes, anomaly_scores, distances, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))
    return scores, signals, window_sizes, anomaly_scores, distances


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    # parse the input arguments
    parser = argparse.ArgumentParser(description='Dash Startup Script.')
    parser.add_argument("--debug", '-d', nargs='?', type=str2bool, default=None, help='Enable debug mode of the app.')
    parser.add_argument("--port", '-p', default=8050, help='Set the application port.')
    __args = parser.parse_args()
    __debug = __args.debug

    # define the debug
    if __debug is None:
        __debug= APPLICATION_LEVEL == Level.DEBUG

    # deactivate the flask logger if __debug is false
    # https://community.plotly.com/t/suppress-dash-server-posts-to-console/8855/2
    if not __debug:
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

    # log some versions
    logger.info(f"Dash version: {dash.__version__}.")
    logger.info(f"Plotly version: {plotly.__version__}.")

    # create the data folder if not already there
    if not os.path.isdir(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)

    # start the application
    logger.info(f"Running Dash Main Page with level {APPLICATION_LEVEL=} and {__debug=}.")
    app.run(debug=__debug, port=__args.port)
