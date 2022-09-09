import tensorflow as tf
from tensorflow.keras.models import Model
from sources.classes.base_model import BaseModel
from tensorflow.keras.layers import Dense, Reshape


class MobileNetV2T(BaseModel):

    def __init__(self, hp, hp_range):
        """
        Class for preparing MobileNetV2 for training

        :param hp: parameter for keras tuner (necessary for RandomSearch)
        :param hp_range: hyperparameters range for RandomSearch
        """
        super().__init__(hp, hp_range)
        self.no_of_layers = 156
        if self.not_trainable_number_of_layers >= self.no_of_layers:
            raise BaseException(f'not_trainable_no_of_layers hyperparameters must be less than {self.no_of_layers} for MobileNetV2')
        self.model = tf.keras.applications.mobilenet_v2.MobileNetV2()
        mobile_net_layers = self.model.layers[self.no_of_layers-1].output
        output = Dense(units=5, activation='softmax')(mobile_net_layers)
        output = Reshape((-1,), input_shape=(1, 1, 5,))(output)
        self.model = Model(inputs=self.model.input, outputs=output)
        for layer in self.model.layers[:self.not_trainable_number_of_layers]:
            layer.trainable = False
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                layer.rate = self.dropout_rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
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

