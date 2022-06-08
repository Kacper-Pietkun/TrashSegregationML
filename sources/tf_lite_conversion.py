import constants
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.models import Model
from dataset_handler import ImagesDataset
from sklearn.metrics import accuracy_score


def load_dataset(name):
    dir_path = os.path.join(constants.ROOT_DIR, constants.PATH_LOCAL_SAVED_DATASET)
    full_path = os.path.join(dir_path, name)
    dataset = ImagesDataset.load_saved_dataset(full_path)
    return dataset


def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size


def convert_bytes(size, unit=None):
    if unit == 'KB':
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == 'MB':
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    return print('File size: ' + str(size) + ' bytes')


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # dataset_public = load_dataset(constants.DATASET_PUBLIC)
    dataset_users = load_dataset(constants.DATASET_USERS)
    # dataset_crawler = load_dataset(constants.DATASET_CRAWLER)
    dataset_similar = load_dataset(constants.DATASET_SIMILAR)

    X_test = dataset_users.data['images']
    y_test = dataset_users.data['labels']

    X_train = dataset_similar.data['images']
    y_train = dataset_similar.data['labels']

    HP_EPOCHS = 5
    HP_MN_NOL = 87
    HP_MN_NTNOL = 64
    HP_LR = 0.001
    HP_DR = 0.05
    HP_BS = 64

    mobile_net = tf.keras.applications.mobilenet.MobileNet()
    mobile_net_layers = mobile_net.layers[HP_MN_NOL].output
    output = Dense(units=5, activation='softmax')(mobile_net_layers)
    output = Reshape((-1,), input_shape=(1, 1, 5,))(output)
    model = Model(inputs=mobile_net.input, outputs=output)
    model.summary()
    for layer in model.layers[:HP_MN_NTNOL]:
        layer.trainable = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = HP_DR
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HP_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    training_history = model.fit(x=X_train,
                                 y=y_train,
                                 batch_size=HP_BS,
                                 epochs=HP_EPOCHS,
                                 verbose=1)

    standard_model_name = 'tf_model.h5'
    standard_path = os.path.join(constants.ROOT_DIR, constants.PATH_BEST_MODELS, standard_model_name)
    model.save(standard_path)
    model = tf.keras.models.load_model(standard_path)
    convert_bytes(get_file_size(standard_path))
    convert_bytes(get_file_size(standard_path), 'MB')
    loss, acc = model.evaluate(X_test, y_test, batch_size=HP_BS)
    print("\nStandard model test set evaluation")
    print(f'loss = {loss}, accuracy = {acc}')

    tf_lite_model_name = 'tf_lite_model.tflite'
    tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = tf_lite_converter.convert()
    lite_path = os.path.join(constants.ROOT_DIR, constants.PATH_BEST_MODELS, tf_lite_model_name)
    open(lite_path, 'wb').write(tflite_model)
    convert_bytes(get_file_size(lite_path))
    convert_bytes(get_file_size(lite_path), 'MB')
    tflite_file_size = get_file_size(lite_path)

    interpreter = tf.lite.Interpreter(model_path=lite_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(input_details[0]['index'], (215, 224, 224, 3))
    interpreter.resize_tensor_input(output_details[0]['index'], (215, 5))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], X_test)
    interpreter.invoke()
    tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
    prediction_classes = np.argmax(tflite_model_predictions, axis=1)
    acc = accuracy_score(prediction_classes, y_test)
    print("\nTF Lite model test set evaluation")
    print(f'accuracy = {acc}')

    tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tf_lite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_lite_converter.target_spec.supported_types = [tf.float16]
    tflite_model = tf_lite_converter.convert()
    tf_lite_float_16_model_name = 'tf_lite_float_16_model.tflite'
    lite_16_path = os.path.join(constants.ROOT_DIR, constants.PATH_BEST_MODELS, tf_lite_float_16_model_name)
    open(lite_16_path, 'wb').write(tflite_model)
    convert_bytes(get_file_size(lite_16_path), 'MB')

    interpreter = tf.lite.Interpreter(model_path=lite_16_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(input_details[0]['index'], (215, 224, 224, 3))
    interpreter.resize_tensor_input(output_details[0]['index'], (215, 5))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], X_test)
    interpreter.invoke()
    tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
    prediction_classes = np.argmax(tflite_model_predictions, axis=1)
    acc = accuracy_score(prediction_classes, y_test)
    print("\nTF Lite 16 model test set evaluation")
    print(f'accuracy = {acc}')

    tf_lite_optimized_model_name = 'tf_lite_optimized_model.tflite'
    lite_optimized_path = os.path.join(constants.ROOT_DIR, constants.PATH_BEST_MODELS, tf_lite_optimized_model_name)
    tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tf_lite_converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = tf_lite_converter.convert()
    open(lite_optimized_path, 'wb').write(tflite_model)
    convert_bytes(get_file_size(lite_optimized_path), 'MB')

    interpreter = tf.lite.Interpreter(model_path=lite_optimized_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(input_details[0]['index'], (215, 224, 224, 3))
    interpreter.resize_tensor_input(output_details[0]['index'], (215, 5))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], X_test)
    interpreter.invoke()
    tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
    prediction_classes = np.argmax(tflite_model_predictions, axis=1)
    acc = accuracy_score(prediction_classes, y_test)
    print("\nTF Lite optimized model test set evaluation")
    print(f'accuracy = {acc}')
