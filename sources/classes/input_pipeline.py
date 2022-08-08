# Based on https://www.tensorflow.org/tutorials/load_data/images?hl=en#using_tfdata_for_finer_control
import os
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class InputPipeline:

    def __init__(self, directory_path, batch_size, image_width, image_height):
        """
        InputPipeline is a helper function to take advantage of tf.data.dataset API. I decided to use it because it
        allows to easily handle big amount of data (when whole data doesn't fit in the memory). With dataset API
        iteration happens in a streaming fashion, so the full dataset does not need to fit into memory.

        :param directory_path: path to the directory with images
        :param batch_size: size of the batch that will be used for model training
        :param image_width: width of images in the dataset
        :param image_height: height of images in the dataset
        """
        directory_path = pathlib.Path(directory_path)
        self.dataset = tf.data.Dataset.list_files(str(directory_path/'*/*'), shuffle=False)
        self.dataset_name = str(directory_path).split("\\")[-1]
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

        self.image_count_whole = len(list(directory_path.glob('*/*.jpg')))
        self.dataset = self.dataset.shuffle(self.image_count_whole, reshuffle_each_iteration=False)
        self.class_names = np.array(sorted([item.name for item in directory_path.glob('*')]))
        self.image_count_by_class = np.array([item for item in np.array([len(list(directory_path.glob(class_name + "/*.jpg"))) for class_name in self.class_names])])
        # create a dataset of image, label pairs
        self.map(self.process_path)
        self.configure_for_performance()

    def concatenate_pipelines(self, pipeline2):
        """
        Concatenate two pipelines - datasets (tf.data.dataset)

        :param pipeline2: pipeline that will be concatenated with this pipeline
        """
        self.dataset = self.dataset.concatenate(pipeline2.dataset)
        self.dataset_name += f" - {pipeline2.dataset_name}"
        if self.batch_size != pipeline2.batch_size or\
                self.image_width != pipeline2.image_width or\
                self.image_height != pipeline2.image_height or\
                self.class_names.sort() != pipeline2.class_names.sort():
            raise Exception("Cannot concatenate two pipelines with different properties")
        self.image_count_whole += pipeline2.image_count_whole
        self.dataset = self.dataset.shuffle(self.image_count_whole, reshuffle_each_iteration=False)
        self.image_count_by_class = list(map(lambda a, b: a + b, self.image_count_by_class, pipeline2.image_count_by_class))

    def print_text_info_about_dataset(self):
        """
        Prints how many images belongs to each class in a dataset
        """
        print(f"******* {self.dataset_name} *******")
        for class_name, count in zip(self.class_names, self.image_count_by_class):
            print(f"{class_name}: {count}")
        print()

    def plot_first_nine_images(self):
        """
        Plots nine images from dataset (3x3) - three rows and three columns
        """
        image_batch, label_batch = next(iter(self.dataset))
        plt.figure(figsize=(10, 7))
        plt.suptitle(self.dataset_name, size=16)
        for i in range(0, 9):
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(image_batch[i].numpy().astype("uint8"))
            label = label_batch[i]
            ax.set_title(self.class_names[label])
            ax.axis("off")
        plt.show()

    def process_path(self, file_path):
        """
        Converts a file path to an (img, label) pair

        :param file_path: path to an image
        """
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.class_names
        label = tf.argmax(one_hot)

        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.image_height, self.image_width])
        return img, label

    def configure_for_performance(self):
        """
        Use buffered prefetching so you can yield data from disk without having I/O become blocking.
        - Dataset.cache keeps the images in memory after they're loaded off disk during the first epoch.
        - Dataset.prefetch overlaps data preprocessing and model execution while training.
        """
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.shuffle(buffer_size=1000)
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def map(self, func):
        """
        Wrapper around tf.data.Dataset.map function. Adjusted to this class

        :param func: function that will be applied to the dataset
        """
        self.dataset = self.dataset.map(func, num_parallel_calls=tf.data.AUTOTUNE)
