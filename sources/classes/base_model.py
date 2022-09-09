class BaseModel:

    def __init__(self, hp, hp_range):
        """
        Class for preparing MobileNet for training

        :param hp: parameter for keras tuner (necessary for RandomSearch)
        :param hp_range: hyperparameters range for RandomSearch
        """
        self.not_trainable_number_of_layers = hp.Int("not_trainable_no_of_layers",
                                                     hp_range["not_trainable_number_of_layers"]["min"],
                                                     hp_range["not_trainable_number_of_layers"]["max"],
                                                     hp_range["not_trainable_number_of_layers"]["step"])
        self.dropout_rate = hp.Float("dropout",
                                     hp_range["dropout_rate"]["min"],
                                     hp_range["dropout_rate"]["max"],
                                     hp_range["dropout_rate"]["step"])
        self.learning_rate = hp.Choice("lr", hp_range["learning_rate"]["choices"])
        self.epochs = hp.Fixed("epochs", hp_range["epochs"]["fixed"])
        self.batch_size = hp.Fixed("batch_size", hp_range["batch_size"]["fixed"])
