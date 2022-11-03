import keras_tuner

EPOCHS = 2
BATCH_SIZE = 8

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

TRAIN_LENGTH = 1034
EVAL_LENGTH = 128

INPUT_IMG_SIZE = 224

def get_hyperparameters(hyperparameters)->keras_tuner.HyperParameters:
    hp_set = keras_tuner.HyperParameters()

    for hp in hyperparameters:
        hp_set.Choice(
            hp, hyperparameters[hp]["values"], default=hyperparameters[hp]['default']
        )

    return hp_set