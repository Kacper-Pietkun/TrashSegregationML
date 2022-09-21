# -----------------------------------------------------------
# Train model and save it
# -----------------------------------------------------------

import os
from sources.scripts import constants as cs
from sources.classes.random_search import RandomSearch


if __name__ == "__main__":
    args = [{
        'gpu': True,
        'log_dir_name': 'no_log',
        'save_model_dir_name': 'my_model.h5',
        'model': 'MobileNetV2',
        # 'train_data': ['augmented_similar', 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'],
        'train_data': ['augmented_similar'],
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'max_trials': 1,
        'hp_range': {
            'alpha':                            {'fixed': 1.0},
            'epochs':                           {'fixed': 100},
            'batch_size':                       {'fixed': 32},
            'learning_rate_top':                {'fixed': 1e-2},
            'learning_rate_whole':              {'fixed': 1e-5},
            'freeze_layers':                    {'fixed': False},
            'not_trainable_number_of_layers':   {'fixed': 151},
            'dropout_rate':                     {'fixed': 0.55},
        }
    }]

    for arg in args:
        training = RandomSearch(arg)
        training.start(save_last_model=True, log_to_tensorboard=False)
