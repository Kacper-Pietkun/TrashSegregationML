# -----------------------------------------------------------
# Training model based on given model with given hyperparameters
# -----------------------------------------------------------

from sources.scripts import constants as cs
from sources.classes.training import Training

if __name__ == "__main__":
    args = {
        'gpu': True,
        'log_dir_name': '4_aug_train_similar_test_aug_users_aftere_refactor',
        'model': 'MobileNet',
        'train_data': ['augmented_similar'],
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

    training = Training(args)
    training.start()
