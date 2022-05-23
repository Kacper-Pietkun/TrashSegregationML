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


if __name__ == "__main__":
    print('load public dataset...')
    dataset_public = load_dataset(constants.PATH_RAW_DATASET_PUBLIC)
    print('save public dataset...')
    save_dataset(dataset_public, constants.PATH_LOCAL_SAVED_DATASET, constants.DATASET_PUBLIC)

    print('load users dataset...')
    dataset_users = load_dataset(constants.PATH_RAW_DATASET_USERS)
    print('save users dataset...')
    save_dataset(dataset_users, constants.PATH_LOCAL_SAVED_DATASET, constants.DATASET_USERS)

    print('load crawler dataset...')
    dataset_crawler = load_dataset(constants.PATH_RAW_DATASET_CRAWLER)
    print('save crawler dataset...')
    save_dataset(dataset_crawler, constants.PATH_LOCAL_SAVED_DATASET, constants.DATASET_CRAWLER)
