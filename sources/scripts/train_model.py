# -----------------------------------------------------------
# Training model based on given model with given hyperparameters
# -----------------------------------------------------------

import os
from sources.scripts import constants as cs
from sources.classes.training import Training

if __name__ == "__main__":

    args = []

    args6 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNetV2', '2_aug_train_similar'),
        'model': 'MobileNetV2',
        'train_data': ['augmented_similar'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'hp': {
            'epochs': [5],
            'base_number_of_layers': [152],
            'not_trainable_number_of_layers': [100, 132, 152],
            'learning_rate': [0.001, 0.01, 0.1],
            'dropout_rate': [0.05],
            'batch_size': [32, 64]
        }
    }
    args.append(args6)

    args7 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNetV2', '3_aug_train_similar_taco'),
        'model': 'MobileNetV2',
        'train_data': ['augmented_similar', 'augmented_taco'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'hp': {
            'epochs': [5],
            'base_number_of_layers': [152],
            'not_trainable_number_of_layers': [100, 132, 152],
            'learning_rate': [0.001, 0.01, 0.1],
            'dropout_rate': [0.05],
            'batch_size': [32, 64]
        }
    }
    args.append(args7)

    args8 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNetV2', '4_aug_train_similar_garythung'),
        'model': 'MobileNetV2',
        'train_data': ['augmented_similar', 'augmented_garythung'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'hp': {
            'epochs': [5],
            'base_number_of_layers': [152],
            'not_trainable_number_of_layers': [100, 132, 152],
            'learning_rate': [0.001, 0.01, 0.1],
            'dropout_rate': [0.05],
            'batch_size': [32, 64]
        }
    }
    args.append(args8)

    args9 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNetV2', '5_aug_train_similar_search_engine'),
        'model': 'MobileNetV2',
        'train_data': ['augmented_similar', 'augmented_search_engine'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'hp': {
            'epochs': [5],
            'base_number_of_layers': [152],
            'not_trainable_number_of_layers': [100, 132, 152],
            'learning_rate': [0.001, 0.01, 0.1],
            'dropout_rate': [0.05],
            'batch_size': [32, 64]
        }
    }
    args.append(args9)

    args10 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNetV2', '6_aug_train_all'),
        'model': 'MobileNetV2',
        'train_data': ['augmented_similar', 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'hp': {
            'epochs': [5],
            'base_number_of_layers': [152],
            'not_trainable_number_of_layers': [100, 132, 152],
            'learning_rate': [0.001, 0.01, 0.1],
            'dropout_rate': [0.05],
            'batch_size': [32, 64]
        }
    }
    args.append(args10)

    args1 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNet', '5_aug_train_similar_taco_search_engine'),
        'model': 'MobileNet',
        'train_data': ['augmented_similar', 'augmented_taco', 'augmented_search_engine'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'hp': {
            'epochs': [5],
            'base_number_of_layers': [87],
            'not_trainable_number_of_layers': [16, 32, 64],
            'learning_rate': [0.00001, 0.0001, 0.001],
            'dropout_rate': [0.05],
            'batch_size': [32, 64]
        }
    }
    args.append(args1)

    args2 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNet', '6_aug_TRAIN_all'),
        'model': 'MobileNet',
        'train_data': ['augmented_similar', 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'hp': {
            'epochs': [5],
            'base_number_of_layers': [87],
            'not_trainable_number_of_layers': [16, 32, 64],
            'learning_rate': [0.00001, 0.0001, 0.001],
            'dropout_rate': [0.05],
            'batch_size': [32, 64]
        }
    }
    args.append(args2)

    args4 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNet', '7_aug_TRAIN_similar_search_engine'),
        'model': 'MobileNet',
        'train_data': ['augmented_similar', 'augmented_search_engine'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'hp': {
            'epochs': [5],
            'base_number_of_layers': [87],
            'not_trainable_number_of_layers': [16, 32, 64],
            'learning_rate': [0.00001, 0.0001, 0.001],
            'dropout_rate': [0.05],
            'batch_size': [32, 64]
        }
    }
    args.append(args4)

    args5 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNet', '8_aug_TRAIN_similar_garythung'),
        'model': 'MobileNet',
        'train_data': ['augmented_similar', 'augmented_garythung'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'hp': {
            'epochs': [5],
            'base_number_of_layers': [87],
            'not_trainable_number_of_layers': [16, 32, 64],
            'learning_rate': [0.00001, 0.0001, 0.001],
            'dropout_rate': [0.05],
            'batch_size': [32, 64]
        }
    }
    args.append(args5)

    for arg in args:
        training = Training(arg)
        training.start()
