# -----------------------------------------------------------
# Training model based on MobileNet with given hyperparameters
# -----------------------------------------------------------
import constants
import os
import tensorflow as tf
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
        optimizer=tf.keras.optimizers.Adam(lr=hyperparameters[HP_LR]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(model, hyperparameters):
    log_dir = os.path.join(constants.ROOT_DIR, constants.PATH_TENSORBOARD_LOGS)
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


if __name__ == "__main__":
    dataset_public = load_dataset(constants.DATASET_PUBLIC)
    dataset_users = load_dataset(constants.DATASET_USERS)
    dataset_crawler = load_dataset(constants.DATASET_CRAWLER)

    # training set contain only images from public datasets
    X_train = dataset_public.data['images']
    y_train = dataset_public.data['labels']

    # actual images from users are distributed equally among validation and test sets
    (X_val, X_test), (y_val, y_test) = dataset_users.hashed_train_test_split(0.5)

    #with tf.device('/gpu:0'):
    session_num = 1

    HP_EPOCHS = hp.HParam("epochs", hp.Discrete([5]))
    HP_MN_NOL = hp.HParam("MobileNet-number_of_layers", hp.Discrete([77, 82, 87]))  # out of 91
    HP_MN_NTNOL = hp.HParam("MobileNet-not_trainable_number_of_layers", hp.Discrete([16, 32, 64]))  # out of 91
    HP_LR = hp.HParam("learning_rate", hp.Discrete([0.00001, 0.0001, 0.001]))
    HP_DR = hp.HParam("dropout_rate", hp.Discrete([0.05, 0.01]))
    HP_BS = hp.HParam("batch_size", hp.Discrete([16, 32, 64]))

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
                            model = create_model(hyperparameters)
                            train_model(model, hyperparameters)
                            session_num += 1
