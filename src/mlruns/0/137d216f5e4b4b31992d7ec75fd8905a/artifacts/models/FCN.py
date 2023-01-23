import tensorflow.keras as keras


def FCN_model(
    number_of_classes: int, input_length: int = None, parameters=1
) -> keras.models.Model:
    def inner(input_layer: keras.layers.Input = keras.layers.Input((input_length, 1))):
        conv1 = keras.layers.Conv1D(
            filters=int(128 * parameters),
            kernel_size=11,
            padding="same",
            input_shape=(None, 1),
        )(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation="relu")(conv1)

        conv2 = keras.layers.Conv1D(
            filters=int(256 * parameters), kernel_size=5, padding="same"
        )(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation("relu")(conv2)

        conv3 = keras.layers.Conv1D(
            int(128 * parameters), kernel_size=3, padding="same"
        )(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation("relu")(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(number_of_classes, activation="softmax")(
            gap_layer
        )
        return output_layer

    return inner
