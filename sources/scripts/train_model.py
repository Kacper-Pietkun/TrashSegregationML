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
        'log_dir_name': os.path.join('MobileNetV2', '7_aug_train_similar'),
        'model': 'MobileNetV2',
        'train_data': ['augmented_similar'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'hp': {
            'epochs': [10],
            'base_number_of_layers': [152, 148, 140],
            'not_trainable_number_of_layers': [100, 132, 152],
            'learning_rate': [0.001, 0.01, 0.1, 0.05],
            'dropout_rate': [0.05, 0.1],
            'batch_size': [32, 48, 64]
        }
    }
    args.append(args1)

    args2 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNetV2', '8_aug_train_similar_taco'),
        'model': 'MobileNetV2',
        'train_data': ['augmented_similar', 'augmented_taco'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'hp': {
            'epochs': [10],
            'base_number_of_layers': [152, 148, 140],
            'not_trainable_number_of_layers': [100, 132, 152],
            'learning_rate': [0.001, 0.01, 0.1, 0.05],
            'dropout_rate': [0.05, 0.1],
            'batch_size': [32, 48, 64]
        }
    }
    args.append(args2)

    for arg in args:
        training = Training(arg)
        training.start()
