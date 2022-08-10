import os
from sources.scripts import constants as cs
from sources.classes.mobile_net import MobileNetT
import tensorflow as tf
from sources.classes.input_pipeline import InputPipeline
import multiprocessing
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp
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
        self.hp_epochs = hp.HParam("epochs", hp.Discrete(args['hp']['epochs']))
        self.hp_base_no_of_layers = hp.HParam("base-number-of-layers", hp.Discrete(args['hp']['base_number_of_layers']))
        self.hp_not_trainable_no_of_layers = hp.HParam("MobileNet-not_trainable_number_of_layers", hp.Discrete(args['hp']['not_trainable_number_of_layers']))
        self.hp_learning_rate = hp.HParam("learning-rate", hp.Discrete(args['hp']['learning_rate']))
        self.hp_dropout_rate = hp.HParam("dropout-rate", hp.Discrete(args['hp']['dropout_rate']))
        self.hp_batch_size = hp.HParam("batch-size", hp.Discrete(args['hp']['batch_size']))
        self.all_runs = len(self.hp_epochs.domain.values) * \
                        len(self.hp_base_no_of_layers.domain.values) * \
                        len(self.hp_not_trainable_no_of_layers.domain.values) * \
                        len(self.hp_learning_rate.domain.values) * \
                        len(self.hp_dropout_rate.domain.values) * \
                        len(self.hp_batch_size.domain.values)
        self.model_name = args['model']
        self.train_data_names = args['train_data']
        self.val_data_names = args['val_data']
        self.image_properties = {
            'height': args['image']['height'],
            'width': args['image']['width']
        }
        self.log_dir_name = args['log_dir_name']
        if args['gpu'] is True:
            self.configure_training_for_gpu()

    def configure_training_for_gpu(self):
        """
        Prepare GPU for training
        """
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def get_model(self, model_name, hyperparameters):
        """
        Initializes model with given hyperparameters and returns model class and its object.

        :param model_name: name of the used model
        :param hyperparameters: hyperparameters for given model
        """
        if model_name == "MobileNet":
            mobile_net = MobileNetT(hyperparameters)
            return MobileNetT, mobile_net.model
        elif model_name == "MobileNetV2":
            mobile_net_v2 = MobileNetV2T(hyperparameters)
            return MobileNetV2T, mobile_net_v2.model
        raise Exception(f"Model {model_name} is not recognized")

    def get_actual_input_pipeline(self, dataset_path, batch_size, image_properties):
        """
        Create and return input pipeline for one dataset basing on given path

        :param dataset_path: path to the dataset
        :param batch_size: batch size for the training
        :param image_properties: dictionary with image's width and height
        """
        path = os.path.join(cs.ROOT_DIR, dataset_path)
        pipeline = InputPipeline(path, batch_size, image_properties['width'], image_properties['height'])
        return pipeline

    def get_input_pipeline(self, data_names, batch_size, image_properties):
        """
        1. Dispatch names of datasets
        2. Create input pipelines for each dataset
        3. Concatenate all of input pipelines, so there is one main input pipeline

        :param data_names: list of datasets' names
        :param batch_size: batch size for the training
        :param image_properties: dictionary with image's width and height
        """
        input_pipelines = []
        for data_name in data_names:
            if data_name == "raw_garythung":
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_RAW_DATASET_GARYTHUNG, batch_size,
                                                                      image_properties))
            elif data_name == 'augmented_garythung':
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_AUGMENTED_DATASET_GARYTHUNG, batch_size,
                                                                      image_properties))
            elif data_name == "raw_search_engine":
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_RAW_DATASET_SEARCH_ENGINE, batch_size,
                                                                      image_properties))
            elif data_name == 'augmented_search_engine':
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_AUGMENTED_DATASET_SEARCH_ENGINE, batch_size,
                                                                      image_properties))
            elif data_name == "raw_similar":
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_RAW_DATASET_SIMILAR, batch_size,
                                                                      image_properties))
            elif data_name == 'augmented_similar':
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_AUGMENTED_DATASET_SIMILAR, batch_size,
                                                                      image_properties))
            elif data_name == "raw_taco":
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_RAW_DATASET_TACO, batch_size,
                                                                      image_properties))
            elif data_name == 'augmented_taco':
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_AUGMENTED_DATASET_TACO, batch_size,
                                                                      image_properties))
            elif data_name == "raw_user":
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_RAW_DATASET_USER, batch_size,
                                                                      image_properties))
            elif data_name == 'augmented_user':
                input_pipelines.append(self.get_actual_input_pipeline(cs.PATH_AUGMENTED_DATASET_USER, batch_size,
                                                                      image_properties))
        for i in range(1, len(input_pipelines)):
            input_pipelines[0].concatenate_pipelines(input_pipelines[i])
        return input_pipelines[0]

    def run_training(self, hyperparameters, model_name, train_data_names, val_data_names, image_properties, log_dir_name):
        """
        Carry out one training process (one from all of the combinations, that are run by start() function)

        :param hyperparameters: hyperparameters for model and training process
        :param model_name: name of the model that will be trained
        :param train_data_names: list of datasets' names that will be used for training
        :param val_data_names: list of datasets' names that will be used for validation
        :param image_properties: dictionary with image's width and height
        :param log_dir_name: name of the dir where logs will be held
        """
        input_pipeline_train = self.get_input_pipeline(train_data_names, hyperparameters['batch_size'], image_properties)
        input_pipeline_val = self.get_input_pipeline(val_data_names, hyperparameters['batch_size'], image_properties)
        model_class, model = self.get_model(model_name, hyperparameters)
        input_pipeline_train.map(model_class.preprocess_input_pipeline)
        input_pipeline_val.map(model_class.preprocess_input_pipeline)
        self.train_model(model, hyperparameters, input_pipeline_train, input_pipeline_val, log_dir_name)

    def train_model(self, model, hyperparameters, pipeline_train, pipeline_val, log_dir_name):
        """
        Actual training after creating model and input pipelines

        :param model: model that will be trained
        :param hyperparameters: hyperparameters for model and training process
        :param pipeline_train: input pipeline for training
        :param pipeline_val: input pipeline for validation
        :param log_dir_name: name of the dir where logs will be held
        """
        log_dir = os.path.join(cs.ROOT_DIR, cs.PATH_TENSORBOARD_LOGS, log_dir_name, datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir)
        keras_callback = hp.KerasCallback(log_dir, hyperparameters)
        training_history = model.fit(pipeline_train.dataset,
                                     validation_data=pipeline_val.dataset,
                                     epochs=hyperparameters['epochs'],
                                     verbose=1,
                                     callbacks=[tensorboard_callback, keras_callback])

    def start(self):
        """
        Call this function in order to start training process. It will make runs for all hyperparameters combinations.
        It will also prepare model and input pipelines.
        """
        session_num = 1
        for epoch in self.hp_epochs.domain.values:
            for bnol in self.hp_base_no_of_layers.domain.values:
                for ntnol in self.hp_not_trainable_no_of_layers.domain.values:
                    for lr in self.hp_learning_rate.domain.values:
                        for dropout_rate in self.hp_dropout_rate.domain.values:
                            for batch_size in self.hp_batch_size.domain.values:
                                hyperparameters = {
                                    'epochs': epoch,
                                    'base_no_of_layers': bnol,
                                    'not_trainable_no_of_layers': ntnol,
                                    'learning_rate': lr,
                                    'dropout_rate': dropout_rate,
                                    'batch_size': batch_size
                                }
                                run_name = "run-%d/%d" % (session_num, self.all_runs)
                                print('--- Starting trial: %s' % run_name)
                                print({h: hyperparameters[h] for h in hyperparameters})
                                process = multiprocessing.Process(target=self.run_training,
                                                                  args=(hyperparameters,
                                                                        self.model_name,
                                                                        self.train_data_names,
                                                                        self.val_data_names,
                                                                        self.image_properties,
                                                                        self.log_dir_name))
                                process.start()
                                process.join()
                                session_num += 1
