import numpy as np


class BaseModel:

    def __init__(self, hp_range):
        """
        Class for preparing MobileNet for training

        :param hp_range: hyperparameters range for RandomSearch
        """

        hp_range_not_trainable = hp_range["not_trainable_number_of_layers"]
        not_trainable_random = np.random.randint(0, len(hp_range_not_trainable["choices"]))
        self.not_trainable_number_of_layers = hp_range_not_trainable["choices"][not_trainable_random]

        hp_range_dropout = hp_range["dropout_rate"]
        dropout_random = np.random.randint(0, (hp_range_dropout["max"] - hp_range_dropout["min"]) / hp_range_dropout["step"] + 1)
        self.dropout_rate = hp_range_dropout["min"] + dropout_random * hp_range_dropout["step"]

        alpha_random = np.random.randint(0, len(hp_range["alpha"]["choices"]))
        self.alpha = hp_range["alpha"]["choices"][alpha_random]

        learning_rate_top_random = np.random.randint(0, len(hp_range["learning_rate_top"]["choices"]))
        self.learning_rate_top = hp_range["learning_rate_top"]["choices"][learning_rate_top_random]

        learning_rate_whole_random = np.random.randint(0, len(hp_range["learning_rate_whole"]["choices"]))
        self.learning_rate_whole = hp_range["learning_rate_whole"]["choices"][learning_rate_whole_random]

        freeze_layers_random = np.random.randint(0, len(hp_range["freeze_layers"]["choices"]))
        self.freeze_layers = hp_range["freeze_layers"]["choices"][freeze_layers_random]

        self.epochs = hp_range["epochs"]["fixed"]
        self.batch_size = hp_range["batch_size"]["fixed"]
