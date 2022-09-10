# -----------------------------------------------------------
# Training model based on given model with given hyperparameters
# -----------------------------------------------------------

import os
from sources.scripts import constants as cs
from sources.classes.training import Training


if __name__ == "__main__":
    args = []

    # args1 = {
    #     'gpu': True,
    #     'log_dir_name': os.path.join('MobileNet', 'test'),
    #     'model': 'MobileNet',
    #     'train_data': ['augmented_similar'],  # , 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'
    #     'val_data': ['raw_user'],
    #     'image': {
    #         'height': cs.IMAGE_HEIGHT,
    #         'width': cs.IMAGE_WIDTH
    #     },
    #     'max_trials': 3,
    #     'executions_per_trial': 1,
    #     'hp_range': {
    #         'alpha':                            {'fixed': 1.0},
    #         'depth_multiplier':                 {'fixed': 1},
    #         'epochs':                           {'fixed': 100},
    #         'batch_size':                       {'fixed': 64},
    #         'learning_rate_top':                {'choices': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]},
    #         'learning_rate_whole':              {'choices': [1e-6, 1e-5]},
    #         'freeze_layers':                    {'choices': [True, False]},
    #         'not_trainable_number_of_layers':   {'choices': [49, 55, 61, 67, 73, 80, 86]},
    #         'dropout_rate':                     {'min': 0,      'max': 0.6,     'step': 0.05},
    #     }
    # }
    # args.append(args1)

    args2 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNetV2', 'test'),
        'model': 'MobileNetV2',
        'train_data': ['augmented_similar'],  # , 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'max_trials': 3,
        'executions_per_trial': 1,
        'hp_range': {
            'alpha': {'fixed': 1.0},
            'depth_multiplier': {'fixed': 1},
            'epochs': {'fixed': 1},
            'batch_size': {'fixed': 64},
            'learning_rate_top': {'choices': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]},
            'learning_rate_whole': {'choices': [1e-6, 1e-5]},
            'freeze_layers': {'choices': [True, False]},
            'not_trainable_number_of_layers': {'choices': [90, 98, 107, 116, 125, 134, 143, 151]},
            'dropout_rate': {'min': 0, 'max': 0.6, 'step': 0.05},
        }
    }
    args.append(args2)

    for arg in args:
        training = Training(arg)
        training.run_training()
