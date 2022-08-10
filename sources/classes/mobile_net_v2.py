import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape


class MobileNetV2T:

    def __init__(self, hyperparameters):
        """
        Class for preparing MobileNetV2 for training

        :param hyperparameters: all the necessary hyperparameters for MobileNetV2
        """
        # number of layers for MobileNetV2 is equal to 156, so hyperparameters like base_no_of_layers and
        # not_trainable_no_of_layers must be smaller than 156
        if hyperparameters['base_no_of_layers'] >= 156 or hyperparameters['not_trainable_no_of_layers'] >= 156:
            raise BaseException('base_no_of_layers and not_trainable_no_of_layers hyperparameters must be less than 156 for MobileNetV2')
        self.model = tf.keras.applications.mobilenet_v2.MobileNetV2()
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
        correctly prepared for MobileNetV2

        :param image: image representing single trash
        :param label: label of the image
        """
        return tf.keras.applications.mobilenet_v2.preprocess_input(image), label

