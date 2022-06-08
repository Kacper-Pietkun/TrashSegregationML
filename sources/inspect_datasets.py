# -----------------------------------------------------------
# Loads saved datasets and prints basic information about those datasets.
# Prints number of images in each class and plots first image from each dataset
# -----------------------------------------------------------
import constants
import os
from dataset_handler import ImagesDataset


if __name__ == "__main__":
    datasets_names = [constants.DATASET_PUBLIC, constants.DATASET_USERS, constants.DATASET_CRAWLER, constants.DATASET_SIMILAR]
    for name in datasets_names:
        path = os.path.join(constants.ROOT_DIR, constants.PATH_LOCAL_SAVED_DATASET)
        full_path = os.path.join(path, name)
        dataset = ImagesDataset.load_saved_dataset(full_path)
        print(name)
        dataset.info()
        print()
        dataset.plot_image(index=0, name=name)
