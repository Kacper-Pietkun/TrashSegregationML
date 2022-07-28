import os
import pickle
import cv2
import numpy as np
import constants
import random
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from matplotlib import pyplot as plt
from zlib import crc32
import tensorflow as tf


# Prepare image dataset to be suitable for MobileNet model
class ImagesDataset:

    def __init__(self, data, image_height, image_width):
        self.data = data
        self.image_height = image_height
        self.image_width = image_width

    @classmethod
    def load_saved_dataset(cls, file_name):
        """
        Initialize dataset by loading one that was saved previously.

        Args:
          file_name (str): name of the saved dataset.
        """
        dir_path = os.path.join(constants.ROOT_DIR, constants.PATH_LOCAL_SAVED_DATASET)
        full_path = os.path.join(dir_path, file_name)
        with open(full_path, 'rb') as file:
            data, image_height, image_width = pickle.load(file)
        return cls(data, image_height, image_width)

    @classmethod
    def load_raw_dataset(cls, dir_path, image_height, image_width, random_state):
        """
        Initialize dataset by loading all of the images from the directory.
        Loaded images are normalized, scaled and saved to an array in RGB format.

        Args:
          dir_path (str): name of the directory where image classes are located.
          image_height (int): height of the image after scaling.
          image_width (int): width of the image after scaling.
          random_state (int): seed to random number generator.
        """
        data = {'images': [], 'labels': [], 'classes': []}
        image_height = image_height
        image_width = image_width
        for dir in os.listdir(dir_path):
            for file in os.listdir(os.path.join(dir_path, dir)):
                image = cls.__load_image(os.path.join(dir_path, dir, file))
                image = cls.__resize_image(image, image_height, image_width)
                image = np.expand_dims(image, axis=0)
                tf.keras.applications.mobilenet.preprocess_input(image)
                data['images'].append(image)
                data['labels'].append(dir)
        class_dict = {img_class: id for id, img_class in enumerate(np.unique(data['labels']))}
        data['labels'] = [class_dict[data['labels'][i]] for i in range(len(data['labels']))]
        data['classes'] = list(class_dict.keys())

        # shuffle dataset
        zipped = list(zip(data['images'], data['labels']))
        random.Random(random_state).shuffle(zipped)
        data['images'], data['labels'] = zip(*zipped)

        # making data suitable for MobileNet
        data['images'] = np.array(data['images']).reshape(-1, 224, 224, 3)
        data['labels'] = np.array(data['labels'])

        return cls(data, image_height, image_width)

    def save_dataset(self, file_name):
        """
        Save dataset to the file.

        Args:
          file_name (str): name of the file where dataset will be saved.
        """
        dir_path = os.path.join(constants.ROOT_DIR, constants.PATH_LOCAL_SAVED_DATASET)
        full_path = os.path.join(dir_path, file_name)
        with open(full_path, 'wb') as file:
            pickle.dump((self.data, self.image_height, self.image_width), file)

    @classmethod
    def __load_image(cls, image_path):
        """
        Loads image, converts it to RGB format, and saves to numpy float array.

        Args:
          image_path (str): path to the file that is going to be loaded.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        image = image.astype('float32')
        return image

    @classmethod
    def __resize_image(cls, image, image_height, image_width):
        """
        Scales image to given sizes.

        Args:
          image (numpy.ndarray): image to scale.
          image_height (int): height of the image after scaling.
          image_width (int): width of the image after scaling.
        """
        return cv2.resize(image, (image_height, image_width),
                          interpolation=cv2.INTER_AREA)

    def stratified_train_test_split(self, test_size, random_state):
        """
        Stratified split of the dataset into train and test sets.

        Args:
          test_size (float): value from 0.0 to 1.0, that specifies size of the test set.
          random_state (int): seed to random number generator.
        """
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_index, test_index in split.split(self.data['images'], self.data['labels']):
            X_train = np.array([self.data['images'][i] for i in train_index]).reshape(-1, 224, 224, 3)
            X_test = np.array([self.data['images'][i] for i in test_index]).reshape(-1, 224, 224, 3)
            y_train = np.array([self.data['labels'][i] for i in train_index])
            y_test = np.array([self.data['labels'][i] for i in test_index])
            return (X_train, X_test), (y_train, y_test)

    def hashed_train_test_split(self, test_size):
        """
        Hashed split of the dataset into train and test sets. Calulates crc32 of an
        image to decide whether it is going to be put in test or train set. This way
        we are sure that one image will always be in the same set (don't need the
        same seed)

        Args:
          test_size (float): value from 0.0 to 1.0, that specifies size of the test set.
        """
        is_in_test_set = list(map(lambda x: crc32(x) & 0xffffffff < test_size * 2 ** 32, self.data['images']))
        X_test = np.array([x for i, x in enumerate(self.data['images']) if is_in_test_set[i] is True]).reshape(-1, 224,
                                                                                                               224, 3)
        y_test = np.array([y for i, y in enumerate(self.data['labels']) if is_in_test_set[i] is True])
        is_in_train_set = [not elem for elem in is_in_test_set]
        X_train = np.array([x for i, x in enumerate(self.data['images']) if is_in_train_set[i] is True]).reshape(-1,
                                                                                                                 224,
                                                                                                                 224, 3)
        y_train = np.array([y for i, y in enumerate(self.data['labels']) if is_in_train_set[i] is True])
        return (X_train, X_test), (y_train, y_test)

    def info(self, y=None):
        """
        Prints info that says how many images from each class there is in a given
        set. If set is not specified, then whole dataset is considered.

        Args:
          y (list of ints): set with class labels.
        """
        if y is None:
            keys = Counter(self.data['labels']).keys()
            values = Counter(self.data['labels']).values()
        else:
            keys = Counter(y).keys()
            values = Counter(y).values()
        for key, value in sorted(zip(keys, values), key=lambda x: x[0]):
            print(self.data['classes'][key], ': ', value)

    def plot_image(self, X=None, y=None, index=0, name=""):
        """
        Plots one image from given set. If set is not specified, then whole dataset
        is considered.

        Args:
          X (list of np.arrays): set with images.
          y (list of ints): set with labels.
          index (int): index of the image.
          name (str): name of the dataset.
        """
        if X is None and y is None:
            X = self.data['images']
            y = self.data['labels']
        elif X is None or y is None:
            raise ValueError('X and y must be both None or not None')
        img = X[index].copy()
        # scale image to range [0, 1] (because it is now in range [-1, 1])
        img = (img + 1.0) / 2.0
        img = np.reshape(img, (224, 224, 3))
        plt.figure(figsize=(5, 5))
        plt.title(name + ': ' + self.data['classes'][y[index]], size=15)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
