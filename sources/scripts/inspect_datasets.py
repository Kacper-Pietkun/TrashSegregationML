# -----------------------------------------------------------
# Plot and print basic information about given datasets
# -----------------------------------------------------------

import os
from sources.scripts import constants as cs
from sources.classes.input_pipeline import InputPipeline


if __name__ == "__main__":
    dataset_paths = [cs.PATH_RAW_DATASET_GARYTHUNG, cs.PATH_RAW_DATASET_SEARCH_ENGINE, cs.PATH_RAW_DATASET_SIMILAR,
                     cs.PATH_RAW_DATASET_TACO, cs.PATH_RAW_DATASET_USER]
    for path in dataset_paths:
        pipeline = InputPipeline(os.path.join(cs.ROOT_DIR, path), cs.CUSTOM_BATCH_SIZE, cs.IMAGE_WIDTH, cs.IMAGE_HEIGHT)
        pipeline.print_text_info_about_dataset()
        pipeline.plot_first_nine_images()
