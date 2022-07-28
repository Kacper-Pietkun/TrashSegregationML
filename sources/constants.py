from pathlib import Path


# Main directory of the project
ROOT_DIR = Path(__file__).parent.parent

# Raw, unprocessed datasets
PATH_RAW_DATASET_GARYTHUNG = r'resources\loaded_datasets\dataset_garythung'
PATH_RAW_DATASET_SEARCH_ENGINE = r'resources\loaded_datasets\dataset_search_engine'
PATH_RAW_DATASET_SIMILAR = r'resources\loaded_datasets\dataset_similar'
PATH_RAW_DATASET_TACO = r'resources\loaded_datasets\dataset_taco'
PATH_RAW_DATASET_TEST = r'resources\loaded_datasets\dataset_test'

# Logs from the latest training
PATH_TENSORBOARD_LOGS = r'tensorboard_logs'

# Location of processed datasets
PATH_LOCAL_SAVED_DATASET = r'resources\saved_datasets'

# Trained models
PATH_BEST_MODELS = r'resources\best_models'

# Names of saved (pickled) datasets
DATASET_GARYTHUNG = 'dataset_garythung'
DATASET_SEARCH_ENGINE = 'dataset_search_engine'
DATASET_SIMILAR = 'dataset_similar'
DATASET_TACO = 'dataset_taco'
DATASET_TEST = 'dataset_test'

DATASET_EXTENSION = '.dat'

# Names of the trash folders
CLASS_BIODEGRADABLE = 'bio'
CLASS_GLASS = 'glass'
CLASS_MIXED = 'mixed'
CLASS_PAPER = 'paper'
CLASS_PLASTIC = 'plastic_metal'

# Image size suitable for MobileNet model
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
