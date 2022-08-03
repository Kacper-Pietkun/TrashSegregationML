# -----------------------------------------------------------
# Plot and print basic information about given datasets
# -----------------------------------------------------------

import os
from sources.scripts import constants as cs
from sources.classes.augmentation_pipeline import AugmentationPipeline


if __name__ == "__main__":
    classes = [cs.CLASS_BIODEGRADABLE, cs.CLASS_GLASS,  cs.CLASS_MIXED,  cs.CLASS_PAPER, cs.CLASS_PLASTIC]

    datasets_raw_paths = [cs.PATH_RAW_DATASET_GARYTHUNG,  cs.PATH_RAW_DATASET_SEARCH_ENGINE,
                          cs.PATH_RAW_DATASET_SIMILAR,  cs.PATH_RAW_DATASET_TACO,
                          cs.PATH_RAW_DATASET_USER]

    datasets_augmented_paths = [cs.PATH_AUGMENTED_DATASET_GARYTHUNG, cs.PATH_AUGMENTED_DATASET_SEARCH_ENGINE,
                                cs.PATH_AUGMENTED_DATASET_SIMILAR, cs.PATH_AUGMENTED_DATASET_TACO,
                                cs.PATH_AUGMENTED_DATASET_USER]

    for raw_path, augmented_path in zip(datasets_raw_paths, datasets_augmented_paths):
        for trash_class in classes:
            pipeline = AugmentationPipeline(trash_class,
                                            os.path.join(cs.ROOT_DIR, raw_path),
                                            os.path.join(cs.ROOT_DIR, augmented_path))
            pipeline.augment()

