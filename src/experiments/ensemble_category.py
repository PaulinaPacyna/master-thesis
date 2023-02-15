from tensorflow import keras

from reading import ConcatenatedDataset
import tensorflow as tf

category = "ECG"
datasets = ConcatenatedDataset().return_datasets_for_category(category=category)
models = []
for dataset in datasets:
    models += [
        keras.models.load_model(
            f"data/models/encoder_same_cat_other_datasets/dest_plain/dataset={dataset}"
        )
    ]
first = keras.layers.Input(models[0].inputs[0].shape[1:])
models[0](first)
models[1](first)
last = tf.keras.layers.Add()([models[0].layers[-1], models[1].layers[-1]])
model = keras.models.Model(inputs=first, outputs=last)
