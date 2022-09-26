# -----------------------------------------------------------
# Perform random search for hyper-parameter optimization
# -----------------------------------------------------------

import os
from sources.scripts import constants as cs
from sources.classes.random_search import RandomSearch


if __name__ == "__main__":
    args = []

    args1 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNet', 'random_proper_tf_4_aug_scale_3_train_all'),
        'save_model_name': 'none',
        'model': 'MobileNet',
        'train_data': ['augmented_similar', 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'max_trials': 5,
        'executions_per_trial': 1,
        'hp_range': {
            'alpha':                            {'choices': [0.25, 0.5, 0.75, 1.0]},
            'epochs':                           {'fixed': 100},
            'batch_size':                       {'fixed': 64},
            'learning_rate_top':                {'choices': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]},
            'learning_rate_whole':              {'choices': [1e-6, 1e-5]},
            'freeze_layers':                    {'choices': [True, False]},
            'not_trainable_number_of_layers':   {'choices': [49, 55, 61, 67, 73, 80, 86]},
            'dropout_rate':                     {'min': 0, 'max': 0.8, 'step': 0.05},
        }
    }
    # args.append(args1)

    args2 = {
        'gpu': True,
        'log_dir_name': os.path.join('MobileNetV2', 'random_proper_tf_5_aug_scale_3_train_all'),
        'save_model_name': 'none',
        'model': 'MobileNetV2',
        'train_data': ['augmented_similar', 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'max_trials': 37,
        'hp_range': {
            'alpha':                            {'choices': [0.35, 0.5, 0.75, 1.0, 1.3, 1.4]},
            'epochs':                           {'fixed': 100},
            'batch_size':                       {'fixed': 32},
            'learning_rate_top':                {'choices': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]},
            'learning_rate_whole':              {'choices': [1e-6, 1e-5]},
            'freeze_layers':                    {'choices': [True, False]},
            'not_trainable_number_of_layers':   {'choices': [90, 98, 107, 116, 125, 134, 143, 151]},
            'dropout_rate':                     {'min': 0, 'max': 0.8, 'step': 0.05},
        }
    }
    args.append(args2)

    for arg in args:
        random_search = RandomSearch(arg)
        random_search.start(save_last_model=False, log_to_tensorboard=True)
