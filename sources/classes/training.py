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
from tensorboard.plugins.hparams import api as hp
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
        self.executions_per_trial = args['executions_per_trial']
        self.log_dir_name = args['log_dir_name']
        if args['gpu'] is True:
            self.configure_training_for_gpu()

    def configure_training_for_gpu(self):
        """
        Prepare GPU for training
        """
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def build_model(self):
        """
        Initializes model with given hyperparameters and returns model class and its object.

        :param hp: hyperparameters for model
        """
        if self.model_name == "MobileNet":
            mobile_net = MobileNetT(self.hp_range)
            return mobile_net
        elif self.model_name == "MobileNetV2":
            mobile_net_v2 = MobileNetV2T(self.hp_range)
            return mobile_net_v2
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
        pipeline = InputPipeline(path, self.hp_range['batch_size']["fixed"],
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


    def train_model(self, pipeline_train, pipeline_val, log_dir_tb):
        """
        Actual training after creating model and input pipelines

        :param pipeline_train: input pipeline for training
        :param pipeline_val: input pipeline for validation
        """
        # Train the top layers
        model = self.build_model()
        base_model = model.base_model
        final_model = model.final_model

        hyperparameters = {
            'epochs': model.epochs,
            'alpha': model.alpha,
            'learning_rate_top': model.learning_rate_top,
            'learning_rate_whole': model.learning_rate_whole,
            'not_trainable_number_of_layers': model.not_trainable_number_of_layers,
            'dropout_rate': model.dropout_rate,
            'batch_size': model.batch_size,
            'freeze_layers': model.freeze_layers
        }
        print(f"Training with:\n"
              f"epochs: {hyperparameters['epochs']}\n"
              f"alpha: {hyperparameters['alpha']}\n"
              f"learning_rate_top: {hyperparameters['learning_rate_top']}\n"
              f"learning_rate_whole: {hyperparameters['learning_rate_whole']}\n"
              f"freeze_layers: {hyperparameters['freeze_layers']}\n"
              f"not_trainable_number_of_layers: {hyperparameters['not_trainable_number_of_layers']}\n"
              f"dropout_rate: {hyperparameters['dropout_rate']}\n"
              f"batch_size: {hyperparameters['batch_size']}\n")

        print("Training top layers")
        model.compile(whole=False)
        tensorboard_callback = TensorBoard(log_dir=log_dir_tb + "top")
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10, verbose=1)
        keras_callback = hp.KerasCallback(log_dir_tb + "top", hyperparameters)
        callbacks = [tensorboard_callback, early_stop_callback, keras_callback]
        final_model.fit(pipeline_train.dataset,
                        validation_data=pipeline_val.dataset,
                        epochs=self.hp_range["epochs"]["fixed"],
                        batch_size=self.hp_range["batch_size"]["fixed"],
                        callbacks=callbacks
                        )

        # Do a round of fine-tuning of the entire model
        print("\nFine-tuning of the entire model")
        base_model.trainable = True
        model.freeze_given_layers()
        model.compile(whole=True)
        tensorboard_callback = TensorBoard(log_dir=log_dir_tb + "whole")
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10, verbose=1)
        keras_callback = hp.KerasCallback(log_dir_tb + "whole", hyperparameters)
        callbacks = [tensorboard_callback, early_stop_callback, keras_callback]
        final_model.fit(pipeline_train.dataset,
                        validation_data=pipeline_val.dataset,
                        epochs=self.hp_range["epochs"]["fixed"],
                        batch_size=self.hp_range["batch_size"]["fixed"],
                        callbacks=callbacks
                        )

    def run_training(self):
        """
        Carry out one training process (one from all of the combinations, that are run by start() function)
        """
        log_dir_tb = os.path.join(cs.ROOT_DIR, cs.PATH_TENSORBOARD_LOGS, self.log_dir_name,
                                  datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        model_class = self.get_model_class()

        input_pipeline_train = self.get_input_pipeline(self.train_data_names)
        input_pipeline_train.map(model_class.preprocess_input_pipeline)

        input_pipeline_val = self.get_input_pipeline(self.val_data_names)
        input_pipeline_val.map(model_class.preprocess_input_pipeline)

        self.train_model(input_pipeline_train, input_pipeline_val, log_dir_tb)

    def start(self):
        """
        Running each training in different process, because this is the only way that I've found that allows clearing
        GPU memory after training
        """
        # Loop for umber of random searches
        for i in range(self.max_trials):
            print(f"\n---- Starting run {i} ----\n")
            process = multiprocessing.Process(target=self.run_training)
            process.start()
            process.join()
