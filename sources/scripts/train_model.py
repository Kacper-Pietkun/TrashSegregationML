# -----------------------------------------------------------
# Train model and save it
# -----------------------------------------------------------

import os
from sources.scripts import constants as cs
from sources.classes.random_search import RandomSearch


if __name__ == "__main__":
    args = [{
        'gpu': True,
        'log_dir_name': os.path.join('MobileNetV2', 'tt'),
        'save_model_name': 'none',
        'model': 'MobileNetV2',
        # 'train_data': ['augmented_similar', 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'],
        'train_data': ['raw_similar'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'max_trials': 0,
        'hp_range': {
            'alpha':                            {'fixed': 1.4},
            'epochs':                           {'fixed': 100},
            'batch_size':                       {'fixed': 32},
            'learning_rate_top':                {'fixed': 0.0001},
            'learning_rate_whole':              {'fixed': 1e-5},
            'freeze_layers':                    {'fixed': False},
            'not_trainable_number_of_layers':   {'fixed': 107},
            'dropout_rate':                     {'fixed': 0.4},
        }
    },
        {
            'gpu': True,
            'log_dir_name': os.path.join('MobileNet', 'tt'),
            'save_model_name': 'none',
            'model': 'MobileNet',
            # 'train_data': ['augmented_similar', 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'],
            'train_data': ['raw_similar'],
            'val_data': ['raw_user'],
            'image': {
                'height': cs.IMAGE_HEIGHT,
                'width': cs.IMAGE_WIDTH
            },
            'max_trials': 0,
            'executions_per_trial': 1,
            'hp_range': {
                'alpha':                            {'fixed': 1.0},
                'epochs':                           {'fixed': 100},
                'batch_size':                       {'fixed': 64},
                'learning_rate_top':                {'fixed': 0.01},
                'learning_rate_whole':              {'fixed': 1e-6},
                'freeze_layers':                    {'fixed': True},
                'not_trainable_number_of_layers':   {'fixed': 49},
                'dropout_rate':                     {'fixed': 0.05},
            }
        },
        {
            'gpu': True,
            'log_dir_name': os.path.join('MobileNetV2', 'keras_augment_similar_3'),
            'save_model_name': 'none',
            'model': 'MobileNetV2',
            # 'train_data': ['augmented_similar', 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'],
            'train_data': ['augmented_similar'],
            'val_data': ['raw_user'],
            'image': {
                'height': cs.IMAGE_HEIGHT,
                'width': cs.IMAGE_WIDTH
            },
            'max_trials': 10,
            'hp_range': {
                'alpha': {'fixed': 1.4},
                'epochs': {'fixed': 100},
                'batch_size': {'fixed': 32},
                'learning_rate_top': {'fixed': 0.0001},
                'learning_rate_whole': {'fixed': 1e-5},
                'freeze_layers': {'fixed': False},
                'not_trainable_number_of_layers': {'fixed': 107},
                'dropout_rate': {'fixed': 0.4},
            }
        },
        {
            'gpu': True,
            'log_dir_name': os.path.join('MobileNet', 'keras_augment_similar_3'),
            'save_model_name': 'none',
            'model': 'MobileNet',
            # 'train_data': ['augmented_similar', 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'],
            'train_data': ['augmented_similar'],
            'val_data': ['raw_user'],
            'image': {
                'height': cs.IMAGE_HEIGHT,
                'width': cs.IMAGE_WIDTH
            },
            'max_trials': 10,
            'executions_per_trial': 1,
            'hp_range': {
                'alpha': {'fixed': 1.0},
                'epochs': {'fixed': 100},
                'batch_size': {'fixed': 64},
                'learning_rate_top': {'fixed': 0.01},
                'learning_rate_whole': {'fixed': 1e-6},
                'freeze_layers': {'fixed': True},
                'not_trainable_number_of_layers': {'fixed': 49},
                'dropout_rate': {'fixed': 0.05},
            }
        }
    ]

    for arg in args:
        training = RandomSearch(arg)
        training.start(save_last_model=False, log_to_tensorboard=True)
