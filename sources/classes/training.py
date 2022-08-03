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
from sources.scripts import constants as cs
import os


class Training:

    def __init__(self, args):
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
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def get_model(self, model_name, hyperparameters):
        if model_name == "MobileNet":
            mobile_net = MobileNetT(hyperparameters)
            return MobileNetT, mobile_net.model
        raise Exception(f"Model {model_name} is not recognized")

    def get_actual_input_pipeline(self, dataset_path, batch_size, image_properties):
        path = os.path.join(cs.ROOT_DIR, dataset_path)
        pipeline = InputPipeline(path, batch_size, image_properties['width'], image_properties['height'])
        return pipeline

    def get_input_pipeline(self, data_names, batch_size, image_properties):
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
        return input_pipelines[0]

    def run_training(self, hyperparameters, model_name, train_data_names, val_data_names, image_properties, log_dir_name):
        input_pipeline_train = self.get_input_pipeline(train_data_names, hyperparameters['batch_size'], image_properties)
        input_pipeline_val = self.get_input_pipeline(val_data_names, hyperparameters['batch_size'], image_properties)
        model_class, model = self.get_model(model_name, hyperparameters)
        input_pipeline_train.map(model_class.preprocess_input_pipeline)
        input_pipeline_val.map(model_class.preprocess_input_pipeline)
        self.train_model(model, hyperparameters, input_pipeline_train, input_pipeline_val, log_dir_name)

    def train_model(self, model, hyperparameters, pipeline_train, pipeline_val, log_dir_name):
        log_dir = os.path.join(cs.ROOT_DIR, cs.PATH_TENSORBOARD_LOGS, log_dir_name, datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir)
        keras_callback = hp.KerasCallback(log_dir, hyperparameters)
        training_history = model.fit(pipeline_train.dataset,
                                     validation_data=pipeline_val.dataset,
                                     epochs=hyperparameters['epochs'],
                                     verbose=1,
                                     callbacks=[tensorboard_callback, keras_callback])

    def start(self):
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
