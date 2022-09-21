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


def get_model_predictions(model, dataset_pipeline):
    y_true = []
    y_pred = []
    for image_batch, label_batch in test_pipeline.dataset:
        batch_pred = model.predict(image_batch)
        y_pred.append(np.argmax(batch_pred, axis=1))
        y_true.append(label_batch)

    correct_labels = tf.concat([item for item in y_true], axis=0)
    predicted_labels = tf.concat([item for item in y_pred], axis=0)
    return correct_labels, predicted_labels


if __name__ == "__main__":
    model = tf.keras.models.load_model(os.path.join(cs.ROOT_DIR, cs.PATH_SAVED_MODELS, 'my_model.h5'))
    test_pipeline = InputPipeline(os.path.join(cs.ROOT_DIR, cs.PATH_RAW_DATASET_USER), cs.CUSTOM_BATCH_SIZE, cs.IMAGE_WIDTH, cs.IMAGE_HEIGHT)
    test_pipeline.map(MobileNetV2T.preprocess_input_pipeline)
    y_true, y_pred = get_model_predictions(model, test_pipeline)
    loss, acc = model.evaluate(test_pipeline.dataset)
    print(f'loss = {loss}\nacc = {acc}')
    plot_pretty_confusion_matrix(y_pred, y_true, test_pipeline.class_names)



