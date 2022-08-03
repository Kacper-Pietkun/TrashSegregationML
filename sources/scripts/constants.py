# -----------------------------------------------------------
# File for constants that are used throughout whole project
# -----------------------------------------------------------

from pathlib import Path


# Main directory of the project
ROOT_DIR = Path(__file__).parent.parent.parent

# Raw, unprocessed datasets
PATH_RAW_DATASET_GARYTHUNG = r'resources\loaded_datasets\dataset_garythung'
PATH_RAW_DATASET_SEARCH_ENGINE = r'resources\loaded_datasets\dataset_search_engine'
PATH_RAW_DATASET_SIMILAR = r'resources\loaded_datasets\dataset_similar'
PATH_RAW_DATASET_TACO = r'resources\loaded_datasets\dataset_taco'
PATH_RAW_DATASET_USER = r'resources\loaded_datasets\dataset_test'

# Raw, augmented datasets
PATH_AUGMENTED_DATASET_GARYTHUNG = r'resources\augmented_datasets\dataset_garythung'
PATH_AUGMENTED_DATASET_SEARCH_ENGINE = r'resources\augmented_datasets\dataset_search_engine'
PATH_AUGMENTED_DATASET_SIMILAR = r'resources\augmented_datasets\dataset_similar'
PATH_AUGMENTED_DATASET_TACO = r'resources\augmented_datasets\dataset_taco'
PATH_AUGMENTED_DATASET_USER = r'resources\augmented_datasets\dataset_test'

# Logs from the latest training
PATH_TENSORBOARD_LOGS = r'tensorboard_logs'

# Trained models
PATH_BEST_MODELS = r'resources\best_models'

# Names of the trash folders
CLASS_BIODEGRADABLE = 'bio'
CLASS_GLASS = 'glass'
CLASS_MIXED = 'mixed'
CLASS_PAPER = 'paper'
CLASS_PLASTIC = 'plastic_metal'

# Image size suitable for MobileNet model
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# Custom values
CUSTOM_BATCH_SIZE = 32
