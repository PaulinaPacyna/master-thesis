from typing import Optional

import tensorflow.keras as keras


def compile_FCN(
    number_of_classes: int,
    initial_learning_rate: float = 1e-4,
    input_lenght: Optional = None,
) -> keras.models.Model:
    input_layer = keras.layers.Input((input_lenght, 1))

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=11, padding="same")(
        input_layer
    )
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation="relu")(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation("relu")(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation("relu")(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(number_of_classes, activation="softmax")(
        gap_layer
    )

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate, decay_steps=3, decay_rate=1
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr_schedule),
        metrics=["accuracy"],
        run_eagerly=True
    )
    return model
