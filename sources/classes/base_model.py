import numpy as np


class BaseModel:

    def __init__(self, hp_range):
        """
        Class for preparing MobileNet for training

        :param hp_range: hyperparameters range for RandomSearch
        """
        self.hp_range = hp_range

        self.dropout_rate = self.get_hp("dropout_rate")
        self.not_trainable_number_of_layers = self.get_hp("not_trainable_number_of_layers")
        self.alpha = self.get_hp("alpha")
        self.freeze_layers = self.get_hp("freeze_layers")
        self.learning_rate_whole = self.get_hp("learning_rate_whole")
        self.learning_rate_top = self.get_hp("learning_rate_top")
        self.epochs = self.get_hp("epochs")
        self.batch_size = self.get_hp("batch_size")

    def get_hp(self, hp_name):
        method_desc = list(self.hp_range[hp_name].keys())[0]
        if method_desc == 'fixed':
            return self.get_fixed(hp_name)
        elif method_desc == 'choices':
            return self.get_choices(hp_name)
        elif method_desc == 'min':
            return self.get_min_max(hp_name)
        raise BaseException(f'cannot dispatch method for {method_desc}')

    def get_fixed(self, hp_name):
        return self.hp_range[hp_name]["fixed"]

    def get_choices(self, hp_name):
        random_index = np.random.randint(0, len(self.hp_range[hp_name]["choices"]))
        return self.hp_range[hp_name]["choices"][random_index]

    def get_min_max(self, hp_name):
        random_index = np.random.randint(0, (self.hp_range[hp_name]["max"] - self.hp_range[hp_name]["min"]) / self.hp_range[hp_name]["step"] + 1)
        return self.hp_range[hp_name]["min"] + random_index * self.hp_range[hp_name]["step"]
