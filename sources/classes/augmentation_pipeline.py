import os
import Augmentor
import numpy as np
from sources.scripts import constants as cs


class AugmentationPipeline:

    def __init__(self, class_name, path_src, path_dst):
        """
        Using Augmentor packet to construct pipeline for data augmentation.

        :param class_name: class directory name
        :param path_src: path to the whole dataset (directory where classes directories are located)
        :param path_dst: path to the whole augmented dataset (directory where classes directories are located)
        """
        self.path_src = os.path.join(path_src, class_name)
        self.path_dst = os.path.join(path_dst, class_name)

        self.pipeline = Augmentor.Pipeline(source_directory=self.path_src, output_directory=self.path_dst,
                                           save_format="JPG")
        self.pipeline.rotate(probability=0.4, max_left_rotation=15, max_right_rotation=15)
        self.pipeline.rotate90(probability=0.5)
        self.pipeline.skew(probability=0.3, magnitude=0.4)
        self.pipeline.shear(probability=0.4, max_shear_right=10, max_shear_left=10)
        self.pipeline.crop_centre(probability=0.5, percentage_area=0.9, randomise_percentage_area=False)
        self.pipeline.flip_left_right(probability=0.5)
        self.pipeline.flip_top_bottom(probability=0.5)
        self.pipeline.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)

        self.images_original_count = self.get_number_of_original_images()
        self.images_number_after_aug = self.get_number_of_augmented_photos(class_name)

    def augment(self):
        """
        Start augmentation of the dataset
        """
        self.pipeline.sample(self.images_number_after_aug)

    def get_number_of_original_images(self):
        """
        Return number of original images (those that are going to be augmented)
        """
        counter = 0
        for path in os.scandir(self.path_src):
            if path.is_file():
                counter += 1
        return counter

    def get_number_of_augmented_photos(self, class_name):
        """
        Estimate target and final number of images after augmentation

        :param class_name: class directory name
        """
        # multiplier_based_on_class = {
        #     'bio': 9,
        #     'glass': 11,
        #     'mixed': 9,
        #     'paper': 8,
        #     'plastic_metal': 5
        # }
        # multiplier = multiplier_based_on_class[class_name]
        # number = self.images_original_count * multiplier if self.images_original_count <= 400 else self.images_original_count
        number = 1500
        return number
