from tensorflow import keras  # pylint: disable=import-error
import tensorflow_addons as tfa


def Encoder_model(
    number_of_classes: int, input_length: int = None, parameters=1
) -> keras.models.Model:
    def inner(input_layer: keras.layers.Input = keras.layers.Input((input_length, 1))):

        # conv block 1
        conv1 = keras.layers.Conv1D(
            filters=int(128 * parameters), kernel_size=5, strides=1, padding="same"
        )(input_layer)
        conv1 = tfa.layers.InstanceNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        # conv block 2
        conv2 = keras.layers.Conv1D(
            filters=int(256 * parameters), kernel_size=11, strides=1, padding="same"
        )(conv1)
        conv2 = tfa.layers.InstanceNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        # conv block 3
        conv3 = keras.layers.Conv1D(
            filters=int(512 * parameters), kernel_size=21, strides=1, padding="same"
        )(conv2)
        conv3 = tfa.layers.InstanceNormalization()(conv3)
        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = keras.layers.Lambda(
            lambda x: x[:, :, : int(256 * parameters)]
        )(conv3)
        attention_softmax = keras.layers.Lambda(
            lambda x: x[:, :, int(256 * parameters) :]
        )(conv3)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])
        # last layer
        dense_layer = keras.layers.Dense(
            units=int(256 * parameters), activation="sigmoid"
        )(multiply_layer)
        dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
        # output layer
        flatten_layer = keras.layers.Flatten()(dense_layer)
        output_layer = keras.layers.Dense(
            units=number_of_classes, activation="softmax"
        )(flatten_layer)

        return output_layer

    return inner
