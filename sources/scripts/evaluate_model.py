import pickle
import os
from sources.scripts import constants as cs
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from sources.classes.input_pipeline import InputPipeline
from sources.classes.mobile_net_v2 import MobileNetV2T
from sklearn.metrics import accuracy_score


def plot_pretty_confusion_matrix(y_pred, y_true, labels):

    cm = confusion_matrix(y_true, y_pred)
    # Normalize our confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create a matrixplot
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.draw()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Label the axes
    ax.set(title='Confusion matrix', xlabel='Predicted label', ylabel='True label')

    # Set x-axis labels to the bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.yaxis.label.set_size(20)
    ax.xaxis.label.set_size(20)
    ax.title.set_size(25)

    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                 horizontalalignment='center',
                 color='white' if cm[i, j] > threshold else 'black',
                 size=10)
    plt.show()


def get_standard_model_predictions(model, test_pipeline):
    y_true = []
    y_pred = []
    for image_batch, label_batch in test_pipeline.dataset:
        batch_pred = model.predict(image_batch)
        y_pred.append(np.argmax(batch_pred, axis=1))
        y_true.append(label_batch)

    correct_labels = tf.concat([item for item in y_true], axis=0)
    predicted_labels = tf.concat([item for item in y_pred], axis=0)
    return correct_labels, predicted_labels


def get_tflite_model_predictions(interpreter, test_pipeline):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_true = []
    y_pred = []
    for image_batch, label_batch in test_pipeline.dataset:
        interpreter.set_tensor(input_details[0]['index'], image_batch)
        interpreter.invoke()
        batch_pred = interpreter.get_tensor(output_details[0]['index'])
        y_pred.append(np.argmax(batch_pred, axis=1))
        y_true.append(label_batch)

    correct_labels = tf.concat([item for item in y_true], axis=0)
    predicted_labels = tf.concat([item for item in y_pred], axis=0)
    return correct_labels, predicted_labels


def get_file_sizes(path):
    file_size_B = get_file_size(path)
    file_size_KB = convert_bytes(file_size_B, 'KB')
    file_size_MB = convert_bytes(file_size_B, 'MB')
    return file_size_B, file_size_KB, file_size_MB


def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size


def convert_bytes(size, unit=None):
    if unit == 'KB':
        return round(size / 1024, 3)
    elif unit == 'MB':
        return round(size / (1024 * 1024), 3)
    return size


def get_model_path(model_name, model_extension):
    return os.path.join(cs.ROOT_DIR, cs.PATH_SAVED_MODELS, model_name + model_extension)


def load_and_evaluate_standard_model(model_name, test_pipeline):
    path = get_model_path(model_name, '.h5')
    file_size_B, file_size_KB, file_size_MB = get_file_sizes(path)
    model = tf.keras.models.load_model(path)

    y_true, y_pred = get_standard_model_predictions(model, test_pipeline)
    loss, acc = model.evaluate(test_pipeline.dataset)
    print("\nStandard model test set evaluation:")
    print(f"Size: {file_size_MB} MB, {file_size_KB} KB, {file_size_B} B")
    print(f'accuracy = {acc}')
    plot_pretty_confusion_matrix(y_pred, y_true, test_pipeline.class_names)


def load_and_evaluate_tflite_model(model_name, optimization, test_pipeline, batch_size):
    if optimization == "standard":
        path = get_model_path(model_name, '_standard.tflite')
    elif optimization == "float_16":
        path = get_model_path(model_name, '_float_16.tflite')
    elif optimization == "tfLite_optimized":
        path = get_model_path(model_name, '_optimized.tflite')
    else:
        raise BaseException(f"cannot handle model with given optimization: {optimization}")
    file_size_B, file_size_KB, file_size_MB = get_file_sizes(path)

    interpreter = tf.lite.Interpreter(model_path=path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]['index'], (batch_size, cs.IMAGE_HEIGHT, cs.IMAGE_WIDTH, 3))
    interpreter.resize_tensor_input(output_details[0]['index'], (batch_size, 5))
    interpreter.allocate_tensors()

    y_true, y_pred = get_tflite_model_predictions(interpreter, test_pipeline)
    acc = accuracy_score(y_pred, y_true)
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    print(f"TF Lite - {optimization} model test set evaluation:")
    print(f"Size: {file_size_MB} MB, {file_size_KB} KB, {file_size_B} B")
    print(f'accuracy = {acc}')
    plot_pretty_confusion_matrix(y_pred, y_true, test_pipeline.class_names)


if __name__ == "__main__":
    batch_size = 1
    test_pipeline = InputPipeline(os.path.join(cs.ROOT_DIR, cs.PATH_RAW_DATASET_USER), batch_size, cs.IMAGE_WIDTH, cs.IMAGE_HEIGHT)
    test_pipeline.map(MobileNetV2T.preprocess_input_pipeline)

    load_and_evaluate_standard_model("model_1", test_pipeline)
    load_and_evaluate_tflite_model("model_1", "standard", test_pipeline, batch_size)
    load_and_evaluate_tflite_model("model_1", "float_16", test_pipeline, batch_size)
    load_and_evaluate_tflite_model("model_1", "tfLite_optimized", test_pipeline, batch_size)







