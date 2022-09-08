import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape


class MobileNetT:

    def __init__(self, hp, hp_range):
        """
        Class for preparing MobileNet for training

        :param hp: parameter for keras tuner (necessary for RandomSearch)
        :param hp_range: hyperparameters range for RandomSearch
        """
        self.no_of_layers = 91
        base_number_of_layers = hp.Int("base_number_of_layers",
                                       hp_range["base_number_of_layers"]["min"],
                                       hp_range["base_number_of_layers"]["max"],
                                       hp_range["base_number_of_layers"]["step"])
        not_trainable_number_of_layers = hp.Int("not_trainable_number_of_layers",
                                                hp_range["not_trainable_number_of_layers"]["min"],
                                                hp_range["not_trainable_number_of_layers"]["max"],
                                                hp_range["not_trainable_number_of_layers"]["step"])
        learning_rate = hp.Float("learning_rate",
                                 hp_range["learning_rate"]["min"],
                                 hp_range["learning_rate"]["max"],
                                 hp_range["learning_rate"]["step"])
        dropout_rate = hp.Float("dropout_rate",
                                hp_range["dropout_rate"]["min"],
                                hp_range["dropout_rate"]["max"],
                                hp_range["dropout_rate"]["step"])

        if base_number_of_layers >= self.no_of_layers or not_trainable_number_of_layers >= self.no_of_layers:
            raise BaseException(f'base_no_of_layers and not_trainable_no_of_layers hyperparameters must be less than {self.no_of_layers} for MobileNet')
        self.model = tf.keras.applications.mobilenet.MobileNet()
        mobile_net_layers = self.model.layers[base_number_of_layers].output
        output = Dense(units=5, activation='softmax')(mobile_net_layers)
        output = Reshape((-1,), input_shape=(1, 1, 5,))(output)
        self.model = Model(inputs=self.model.input, outputs=output)
        for layer in self.model.layers[:not_trainable_number_of_layers]:
            layer.trainable = False
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                layer.rate = dropout_rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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

