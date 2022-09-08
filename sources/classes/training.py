import os
from sources.scripts import constants as cs
from sources.classes.mobile_net import MobileNetT
import keras_tuner
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sources.classes.input_pipeline import InputPipeline
import multiprocessing
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from sources.classes.mobile_net import MobileNetT
from sources.classes.mobile_net_v2 import MobileNetV2T
from sources.scripts import constants as cs
import os


class Training:

    def __init__(self, args):
        """
        Training class is responsible for carrying out process of training a model.
        :param args: dictionary that contains crucial information for training process ie. model name, used datasets
        (for train and validation), hyperparameters, name of the logging directory, whether to use GPU for training,
        image properties (width, heihgt)
        """
        self.hp_range = args["hp_range"]
        self.model_name = args['model']
        self.train_data_names = args['train_data']
        self.val_data_names = args['val_data']
        self.image_properties = {
            'height': args['image']['height'],
            'width': args['image']['width']
        }
        self.max_trials = args['max_trials']
        self.log_dir_name = args['log_dir_name']
        if args['gpu'] is True:
            self.configure_training_for_gpu()

    def configure_training_for_gpu(self):
        """
        Prepare GPU for training
        """
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def build_model(self, hp):
        """
        Initializes model with given hyperparameters and returns model class and its object.

        :param hp: hyperparameters for model
        """
        if self.model_name == "MobileNet":
            mobile_net = MobileNetT(hp, self.hp_range)
            return mobile_net.model
        elif self.model_name == "MobileNetV2":
            mobile_net_v2 = MobileNetV2T(hp, self.hp_range)
            return mobile_net_v2.model
        raise Exception(f"Model {self.model_name} is not recognized")

    def get_model_class(self):
        """
        returns model class basing on its name
        """
        if self.model_name == "MobileNet":
            return MobileNetT
        elif self.model_name == "MobileNetV2":
            return MobileNetV2T
        raise Exception(f"Model {self.model_name} is not recognized")

    def get_actual_input_pipeline(self, dataset_path):
        """
        Create and return input pipeline for one dataset basing on given path

        :param dataset_path: path to the dataset
        """
        path = os.path.join(cs.ROOT_DIR, dataset_path)
        pipeline = InputPipeline(path, self.hp_range['batch_size']['max'],
                                 self.image_properties['width'],
                                 self.image_properties['height'])
        return pipeline

    def get_input_pipeline(self, data_names):
        """
        1. Dispatch names of datasets
        2. Create input pipelines for each dataset
        3. Concatenate all of input pipelines, so there is one main input pipeline

        :param data_names: list of datasets' names
        """
        input_pipelines = []
        for data_name in data_names:
            if data_name == "raw_garythung":
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_RAW_DATASET_GARYTHUNG))
            elif data_name == 'augmented_garythung':
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_AUGMENTED_DATASET_GARYTHUNG))
            elif data_name == "raw_search_engine":
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_RAW_DATASET_SEARCH_ENGINE))
            elif data_name == 'augmented_search_engine':
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_AUGMENTED_DATASET_SEARCH_ENGINE))
            elif data_name == "raw_similar":
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_RAW_DATASET_SIMILAR))
            elif data_name == 'augmented_similar':
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_AUGMENTED_DATASET_SIMILAR))
            elif data_name == "raw_taco":
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_RAW_DATASET_TACO))
            elif data_name == 'augmented_taco':
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_AUGMENTED_DATASET_TACO))
            elif data_name == "raw_user":
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_RAW_DATASET_USER))
            elif data_name == 'augmented_user':
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_AUGMENTED_DATASET_USER))
        for i in range(1, len(input_pipelines)):
            input_pipelines[0].concatenate_pipelines(input_pipelines[i])
        return input_pipelines[0]

    def run_training(self):
        """
        Carry out one training process (one from all of the combinations, that are run by start() function)
        """
        model_class = self.get_model_class()
        input_pipeline_train = self.get_input_pipeline(self.train_data_names)
        input_pipeline_val = self.get_input_pipeline(self.val_data_names)
        input_pipeline_train.map(model_class.preprocess_input_pipeline)
        input_pipeline_val.map(model_class.preprocess_input_pipeline)

        log_dir = os.path.join(cs.ROOT_DIR, cs.PATH_TENSORBOARD_LOGS, self.log_dir_name, "raw",
                               datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        tuner = keras_tuner.RandomSearch(
            self.build_model,
            max_trials=self.max_trials,
            executions_per_trial=1,
            objective="val_accuracy",
            directory=log_dir
        )
        self.train_model(tuner, input_pipeline_train, input_pipeline_val)

    def train_model(self, tuner, pipeline_train, pipeline_val):
        """
        Actual training after creating model and input pipelines

        :param tuner: model that will be trained
        :param pipeline_train: input pipeline for training
        :param pipeline_val: input pipeline for validation
        """
        log_dir = os.path.join(cs.ROOT_DIR, cs.PATH_TENSORBOARD_LOGS, self.log_dir_name,
                               datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir)
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10, verbose=1)
        callbacks = [tensorboard_callback, early_stop_callback]
        tuner.search(
            pipeline_train.dataset,
            validation_data=pipeline_val.dataset,
            epochs=self.hp_range["epochs"],
            batch_size=self.hp_range["batch_size"]["max"],
            callbacks=callbacks
        )
