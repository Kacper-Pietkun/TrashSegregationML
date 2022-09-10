import tensorflow as tf
from tensorflow.keras.models import Model
from sources.classes.base_model import BaseModel
from tensorflow.keras.layers import Dense, Reshape, Input, Dropout
import sources.scripts.constants as cs


class MobileNetV2T(BaseModel):

    def __init__(self, hp_range):
        """
        Class for preparing MobileNet for training. In constructor model is prepared for the first step of training
        that is crucial for transfer learning. We freeze all of the MobileNetV2's layers and add some new trainable
        layers on top.

        :param hp_range: hyperparameters range for RandomSearch
        """
        super().__init__(hp_range)
        self.no_of_layers = 156
        if self.not_trainable_number_of_layers >= self.no_of_layers:
            raise BaseException(f'not_trainable_no_of_layers hyperparameters must be less than {self.no_of_layers} for MobileNetV2')
        self.base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=(cs.IMAGE_WIDTH, cs.IMAGE_HEIGHT, 3),
            alpha=self.alpha,
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
        self.base_model.trainable = False
        inputs = Input(shape=(cs.IMAGE_WIDTH, cs.IMAGE_HEIGHT, 3))
        self.final_model = self.base_model(inputs, training=False)
        self.final_model = Dropout(self.dropout_rate)(self.final_model)
        self.final_model = Dense(units=5, activation='softmax')(self.final_model)
        self.final_model = Reshape((-1,), input_shape=(1, 1, 5,))(self.final_model)
        self.final_model = Model(inputs=inputs, outputs=self.final_model)

    def freeze_given_layers(self):
        """
        If it is decided by the hyperparameters then freeze given number of layers of the base model (MobileNetV2)
        """
        if self.freeze_layers is True:
            for layer in self.base_model.layers[:self.not_trainable_number_of_layers]:
                layer.trainable = False

    def compile(self, whole=True):
        """
        Compilation of the model.

        :param whole: decide which learning rate to use. It is either learning rate for training top layers of the
        model or learning rate for fine-tuneing the model
        """
        self.final_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate_whole if whole is True else self.learning_rate_top),
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

