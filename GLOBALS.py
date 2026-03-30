import enum
import os
import argparse
import logging


# make some enums
class Level(enum.Enum):
    PROD = -1
    INFO = 0
    DEMO = 1
    DEBUG = 2

# define passwords
DELETE_PASSWORD = os.environ.get("CPDASH_DELETE_PASSWORD", "changeme")

# define some information
PAGED = True
DEFAULT_UPLOAD_TEXT = "📤 Please upload your ZIP file"
DEFAULT_UPLOAD_TEXT_SUCCESS = f"✅ Uploaded"

# define a reduced sensor set
MAX_SIGNALS = 800
MOCK_SIGNALS = False

# define the maximum number of signals for the heatmap
MAX_HEATMAP_SIGNALS = 150
MAX_HEATMAP_SELECT_SIGNALS = 40

# define the maximum number of shape selections
# !!!Please do not change this, if you are not absolutely certain that Plotly supports more shapes!!!
# More Information: https://plotly.com/python/performance/
MAX_PLOTLY_SHAPES = 8

# define the maximum number of signals per raw signal plot
RAW_SIGNAL_PLOT_MAXIMUM_NUMBER = 5

# the size of the cache for the cached data (is mostly equivalent to the number of planned users)
# take care: data is stored in memory for every user!
CACHE_SIZE = 1

# SOME FUNCTIONALITY BASED ON THE GLOBAL SETTINGS ----------------------------------------------------------------------

# parse the input arguments
__parser = argparse.ArgumentParser(description='Dash Startup Script.')
__mode_arg = __parser.add_argument('--mode', '-m', default='debug', help=f'Set the application mode. Possible values: {list(ele.name for ele in Level)}')
__port_arg = __parser.add_argument('--port', '-p', default=8050, type=int, help='Set the application port.')
__file_arg = __parser.add_argument('--folder', '-f', default=r'C:\Users\lucas\Data\CP_Anomaly\output_api', help='Set the application folder.')
__args = __parser.parse_args()


# get the application level
__application_level_list = list(ele.name for ele in Level)
__application_level_string = __args.mode.upper()
if __application_level_string not in __application_level_list:
    raise argparse.ArgumentError(__mode_arg, f"Application level must be one of {__application_level_list}. You provided {__application_level_string}.")
APPLICATION_LEVEL = Level[__application_level_string]

# check whether the data folder exists
DATA_FOLDER = __args.folder
if not os.path.isdir(DATA_FOLDER):
    raise argparse.ArgumentError(__file_arg, f"{DATA_FOLDER=} is not a valid directory.")

# get the port
APP_PORT = __args.port


# deactivate the flask logger if we have any low application level
# https://community.plotly.com/t/suppress-dash-server-posts-to-console/8855/2
if APPLICATION_LEVEL.value <= Level.INFO.value:
    __log = logging.getLogger('werkzeug')
    __log.setLevel(logging.ERROR)

# check whether we want to run the application in debug mode
APP_DEBUG = APPLICATION_LEVEL.value >= Level.DEBUG.value