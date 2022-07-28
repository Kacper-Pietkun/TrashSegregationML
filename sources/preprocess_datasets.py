# -----------------------------------------------------------
# Takes raw datasets (images in folders) and preprocess them (scale, normalize, ...)
# and saves preprocessed datasets with pickle to given folders
# -----------------------------------------------------------
import constants
import os
from dataset_handler import ImagesDataset


def load_dataset(path):
    full_path = os.path.join(constants.ROOT_DIR, path)
    dataset = ImagesDataset.load_raw_dataset(full_path,
                                             constants.IMAGE_HEIGHT,
                                             constants.IMAGE_WIDTH,
                                             42)
    return dataset


def save_dataset(dataset, path, dataset_name):
    dir_path = os.path.join(constants.ROOT_DIR, path)
    full_path = os.path.join(dir_path, dataset_name)
    dataset.save_dataset(full_path)


def preprocess_dataset(name, raw_dataset_path, destination_path):
    print(f'load {name} dataset...')
    dataset_similar = load_dataset(raw_dataset_path)
    print(f'save {name} dataset...')
    save_dataset(dataset_similar, destination_path, name + constants.DATASET_EXTENSION)


if __name__ == "__main__":
    preprocess_dataset(constants.DATASET_GARYTHUNG, constants.PATH_RAW_DATASET_GARYTHUNG,
                       constants.PATH_LOCAL_SAVED_DATASET)
    preprocess_dataset(constants.DATASET_SEARCH_ENGINE, constants.PATH_RAW_DATASET_SEARCH_ENGINE,
                       constants.PATH_LOCAL_SAVED_DATASET)
    preprocess_dataset(constants.DATASET_SIMILAR, constants.PATH_RAW_DATASET_SIMILAR,
                       constants.PATH_LOCAL_SAVED_DATASET)
    preprocess_dataset(constants.DATASET_TACO, constants.PATH_RAW_DATASET_TACO,
                       constants.PATH_LOCAL_SAVED_DATASET)
    preprocess_dataset(constants.DATASET_TEST, constants.PATH_RAW_DATASET_TEST,
                       constants.PATH_LOCAL_SAVED_DATASET)

