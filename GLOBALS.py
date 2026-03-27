import enum
import os

# make some enums
class Level(enum.Enum):
    DEBUG = 0
    INFO = 1

# define passwords
DELETE_PASSWORD = os.environ.get("CPDASH_DELETE_PASSWORD", "changeme")

# define some information
DATA_FOLDER = os.environ.get("CPDASH_DATA_PATH", r"C:\Users\lucas\Data\CP_Anomaly\output_api")
APPLICATION_LEVEL = Level.DEBUG
PAGED = True
DEFAULT_UPLOAD_TEXT = "📤 Please upload your ZIP file"
DEFAULT_UPLOAD_TEXT_SUCCESS = f"✅ Uploaded"

# define a reduced sensor set
MAX_SIGNALS = 500

# define the maximum number of shape selections
# !!!Please do not change this, if you are not absolutely certain that Plotly supports more shapes!!!
# More Information: https://plotly.com/python/performance/
MAX_PLOTLY_SHAPES = 8

# define the maximum number of signals per raw signal plot
RAW_SIGNAL_PLOT_MAXIMUM_NUMBER = 5

# the size of the cache for the cached data (is mostly equivalent to the number of planned users)
# take care: data is stored in memory for every user!
CACHE_SIZE = 1

# check whether the data folder exists
if not os.path.isdir(DATA_FOLDER):
    pass
    # raise ValueError(f"{DATA_FOLDER=} is not a valid directory.")
