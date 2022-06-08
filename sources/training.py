# -----------------------------------------------------------
# Training model based on MobileNet with given hyperparameters
# -----------------------------------------------------------
import constants
import os
import tensorflow as tf
import multiprocessing
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp
from dataset_handler import ImagesDataset
from datetime import datetime


def load_dataset(name):
    dir_path = os.path.join(constants.ROOT_DIR, constants.PATH_LOCAL_SAVED_DATASET)
    full_path = os.path.join(dir_path, name)
    dataset = ImagesDataset.load_saved_dataset(full_path)
    return dataset


def create_model(hyperparameters):
    HP_EPOCHS, HP_MN_NOL, HP_MN_NTNOL, HP_LR, HP_DR, HP_BS = list(hyperparameters.keys())
    mobile_net = tf.keras.applications.mobilenet.MobileNet()
    mobile_net_layers=mobile_net.layers[hyperparameters[HP_MN_NOL]].output
    output = Dense(units=5, activation='softmax')(mobile_net_layers)
    output = Reshape((-1,), input_shape=(1, 1, 5,))(output)
    model = Model(inputs=mobile_net.input, outputs=output)
    for layer in model.layers[:hyperparameters[HP_MN_NTNOL]]:
        layer.trainable = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = hyperparameters[HP_DR]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters[HP_LR]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(model, hyperparameters, train, val, test):
    HP_EPOCHS, HP_MN_NOL, HP_MN_NTNOL, HP_LR, HP_DR, HP_BS = list(hyperparameters.keys())
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    log_dir = os.path.join(constants.ROOT_DIR, constants.PATH_TENSORBOARD_LOGS, 'similar_dataset_train')
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    keras_callback = hp.KerasCallback(log_dir, hyperparameters)
    training_history = model.fit(x=X_train,
                                 y=y_train,
                                 validation_data=(X_val, y_val),
                                 batch_size=hyperparameters[HP_BS],
                                 epochs=hyperparameters[HP_EPOCHS],
                                 verbose=1,
                                 callbacks=[tensorboard_callback, keras_callback])


def run_tensorflow(hyperparameters, train, val, test):
    model = create_model(hyperparameters)
    train_model(model, hyperparameters, train, val, test)


if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    dataset_public = load_dataset(constants.DATASET_PUBLIC)
    dataset_users = load_dataset(constants.DATASET_USERS)
    dataset_crawler = load_dataset(constants.DATASET_CRAWLER)
    dataset_similar = load_dataset(constants.DATASET_SIMILAR)

    # training set contain only images from public datasets
    X_test = dataset_users.data['images']
    y_test = dataset_users.data['labels']

    # actual images from users are distributed equally among validation and test sets
    # (X_train, X_val), (y_train, y_val) = dataset_public.stratified_train_test_split(test_size=0.2, random_state=42)

    X_train = dataset_similar.data['images']
    y_train = dataset_similar.data['labels']

    train = X_train, y_train
    val = X_test, y_test
    test = X_test, y_test

    session_num = 1
    HP_EPOCHS = hp.HParam("epochs", hp.Discrete([5]))
    HP_MN_NOL = hp.HParam("MobileNet-number_of_layers", hp.Discrete([87]))  # out of 91
    HP_MN_NTNOL = hp.HParam("MobileNet-not_trainable_number_of_layers", hp.Discrete([16, 32, 64]))  # out of 91
    HP_LR = hp.HParam("learning_rate", hp.Discrete([0.00001, 0.0001, 0.001]))
    HP_DR = hp.HParam("dropout_rate", hp.Discrete([0.05]))
    HP_BS = hp.HParam("batch_size", hp.Discrete([64]))

    all_runs = len(HP_EPOCHS.domain.values) * \
               len(HP_MN_NOL.domain.values) * \
               len(HP_MN_NTNOL.domain.values) * \
               len(HP_LR.domain.values) * \
               len(HP_DR.domain.values) * \
               len(HP_BS.domain.values)

    for epoch in HP_EPOCHS.domain.values:
        for mn_nol in HP_MN_NOL.domain.values:
            for mn_ntnol in HP_MN_NTNOL.domain.values:
                for lr in HP_LR.domain.values:
                    for droput_rate in HP_DR.domain.values:
                        for batch_size in HP_BS.domain.values:
                            hyperparameters = {
                                HP_EPOCHS: epoch,
                                HP_MN_NOL: mn_nol,
                                HP_MN_NTNOL: mn_ntnol,
                                HP_LR: lr,
                                HP_DR: droput_rate,
                                HP_BS: batch_size
                            }
                            run_name = "run-%d/%d" % (session_num, all_runs)
                            print('--- Starting trial: %s' % run_name)
                            print({h.name: hyperparameters[h] for h in hyperparameters})
                            process = multiprocessing.Process(target=run_tensorflow, args=(hyperparameters, train, val, test))
                            process.start()
                            process.join()
                            session_num += 1
