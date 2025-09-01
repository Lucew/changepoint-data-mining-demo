import enum
import os

# make some enums
class Level(enum.Enum):
    DEBUG = 0
    INFO = 1

# define passwords
DELETE_ROOT = "./tmp-data-folder"  # Folder whose contents will be fully removed
DELETE_PASSWORD = os.environ.get("CPDASH_DELETE_PASSWORD", "changeme")

# define some information
DATA_FOLDER = "tmp-data-folder"
APPLICATION_LEVEL = Level.DEBUG
PAGED = True
DEFAULT_UPLOAD_TEXT = "📤 Please upload your ZIP file"
