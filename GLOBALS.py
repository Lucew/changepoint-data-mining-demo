import enum
import os

# make some enums
class Level(enum.Enum):
    DEBUG = 0
    INFO = 1

# define passwords
DELETE_PASSWORD = os.environ.get("CPDASH_DELETE_PASSWORD", "changeme")

# define some information
DATA_FOLDER = os.environ.get("CPDASH_DATA_PATH", "./tmp-data-folder")
APPLICATION_LEVEL = Level.DEBUG
PAGED = True
DEFAULT_UPLOAD_TEXT = "📤 Please upload your ZIP file"
DEFAULT_UPLOAD_TEXT_SUCCESS = f"✅ Uploaded"

# check whether the data folder exists
if not os.path.isdir(DATA_FOLDER):
    raise ValueError(f"{DATA_FOLDER=} is not a valid directory.")
