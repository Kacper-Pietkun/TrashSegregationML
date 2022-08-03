import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape


class MobileNetT:

    def __init__(self, hyperparameters):
        """
        Class for preparing  MobileNet for training

        :param hyperparameters: all the necessary hyperparameters for MobileNet
        """
        # number of layers for MobileNet is equal to 91, so hyperparameters like hp_mn_nol and
        # hp_mn_ntnol must be smaller than 91
        self.model = tf.keras.applications.mobilenet.MobileNet()
        mobile_net_layers = self.model.layers[hyperparameters['base_no_of_layers']].output
        output = Dense(units=5, activation='softmax')(mobile_net_layers)
        output = Reshape((-1,), input_shape=(1, 1, 5,))(output)
        self.model = Model(inputs=self.model.input, outputs=output)
        for layer in self.model.layers[:hyperparameters['not_trainable_no_of_layers']]:
            layer.trainable = False
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                layer.rate = hyperparameters['dropout_rate']
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    @staticmethod
    def preprocess_input_pipeline(image, label):
        """
        images from InputPipeline will go through this function. Images are going to be preprocessed so they are
        correctly prepared for MobileNet

        :param image: image representing single trash
        :param label: label of the image
        """
        return tf.keras.applications.mobilenet.preprocess_input(image), label

