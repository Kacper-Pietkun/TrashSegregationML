from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent


# Paths
PATH_RAW_DATASET_USERS = r'resources\loaded_datasets\dataset_from_users'
PATH_RAW_DATASET_PUBLIC = r'resources\loaded_datasets\dataset_public'
PATH_RAW_DATASET_CRAWLER = r'resources\loaded_datasets\dataset_webcrawler'
PATH_TENSORBOARD_LOGS = r'tensorboard_logs'
PATH_LOCAL_SAVED_DATASET = r'resources\saved_datasets'


# Names of saved (pickled) datasets
DATASET_PUBLIC = 'dataset_public.dat'
DATASET_USERS = 'dataset_users.dat'
DATASET_CRAWLER = 'dataset_webcrawler.dat'


# Names of the trash folders
CLASS_BIODEGRADABLE = 'bio'
CLASS_GLASS = 'glass'
CLASS_MIXED = 'mixed'
CLASS_PAPER = 'paper'
CLASS_PLASTIC = 'plastic_metal'


# Image size suitable for MobileNet model
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
