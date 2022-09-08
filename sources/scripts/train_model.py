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
        'log_dir_name': os.path.join('MobileNetV2', 'test'),
        'model': 'MobileNetV2',
        'train_data': ['augmented_similar'],  # , 'augmented_taco', 'augmented_search_engine', 'augmented_garythung'
        'val_data': ['raw_user'],
        'image': {
            'height': cs.IMAGE_HEIGHT,
            'width': cs.IMAGE_WIDTH
        },
        'max_trials': 3,
        'hp_range': {
            'epochs': 2,
            'base_number_of_layers':            {'min': 151,    'max': 155,     'step': 1},
            'not_trainable_number_of_layers':   {'min': 100,    'max': 132,     'step': 1},
            'learning_rate':                    {'min': -6,     'max': -2,      'step': 1},
            'dropout_rate':                     {'min': 0,      'max': 0.6,     'step': 0.05},
            'batch_size':                       {'min': 32,     'max': 64,     'step': 32}
        }
    }
    args.append(args1)

    for arg in args:
        training = Training(arg)
        training.run_training()
