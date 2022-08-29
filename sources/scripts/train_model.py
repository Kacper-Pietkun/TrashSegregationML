# -----------------------------------------------------------
# Training model based on given model with given hyperparameters
# -----------------------------------------------------------

import os
from sources.scripts import constants as cs
from sources.classes.training import Training

if __name__ == "__main__":

    args = []

    args1 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNetV2', '9_aug_train_all'),
        'model': 'MobileNetV2',
        'train_data': ['augmented_similar', 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'hp': {
            'epochs': [10],
            'base_number_of_layers': [155, 154, 153, 152, 151],
            'not_trainable_number_of_layers': [100, 108, 116, 124, 132],
            'learning_rate': [0.001, 0.01, 0.005],
            'dropout_rate': [0.1],
            'batch_size': [32, 48, 64, 100]
        }
    }
    args.append(args1)

    args2 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNet', '9_aug_train_all'),
        'model': 'MobileNet',
        'train_data': ['augmented_similar', 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'hp': {
            'epochs': [5],
            'base_number_of_layers': [89, 88, 87, 86],
            'not_trainable_number_of_layers': [16, 24, 32, 48, 64],
            'learning_rate': [0.00001, 0.0005, 0.0001, 0.005, 0.001, 0.01],
            'dropout_rate': [0.1],
            'batch_size': [32, 48, 64, 100]
        }
    }
    args.append(args2)

    for arg in args:
        training = Training(arg)
        training.start()
