from typing import Tuple

import keras_nlp
from tensorflow import keras
import tensorflow as tf


class ReductionLayer(keras.layers.Layer):
    def __init__(self, axis=1):
        super(ReductionLayer, self).__init__()
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)


class ShiftingLayer(keras.layers.Layer):
    def __init__(self, width=1):
        super(ShiftingLayer, self).__init__()
        self.width = width
        self.amplitude = width ** (1 / 2) * 1.772637204826652153

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        input_data, shifting_weights = inputs
        index_input = tf.reshape(
            tf.range(1, 1 + tf.shape(input_data)[-2], dtype=shifting_weights.dtype),
            (1, 1, -1),
        )
        index_shift = tf.reshape(
            tf.range(1, 1 + shifting_weights.shape[-1], dtype=shifting_weights.dtype),
            (1, -1),
        )
        index_shift += shifting_weights
        scaling_factor = tf.divide(tf.shape(input_data)[-2], shifting_weights.shape[-1])
        index_shift *= tf.cast(scaling_factor, dtype=index_shift.dtype)
        index_shift = tf.expand_dims(index_shift, axis=-1)
        M = index_input - index_shift
        M = tf.exp(-tf.square(M) / self.width) / self.amplitude
        return tf.matmul(M, input_data)


class WarpingLayer(keras.models.Model):
    def __init__(
        self, output_length, number_of_convolutions=64, kernel_size=11, radial_width=1
    ):
        super(WarpingLayer, self).__init__()
        # hyperparameters
        self.output_length = output_length
        self.number_of_convolutions = number_of_convolutions
        self.kernel_size = kernel_size
        self.radial_width = radial_width
        # layers
        self.conv1 = keras.layers.Conv1D(
            filters=self.number_of_convolutions,
            kernel_size=self.kernel_size,
            padding="same",
        )
        self.batch_normalization = keras.layers.BatchNormalization()
        self.softmax = keras.layers.Softmax(axis=1)
        self.sine_position_encoding = keras_nlp.layers.SinePositionEncoding()
        self.dense = keras.layers.Dense(
            units=self.output_length,
            activation="sigmoid",
            initializer=tf.keras.initializers.HeNormal(),
        )
        self.sum = ReductionLayer(axis=1)
        self.shifting_layer = ShiftingLayer(width=self.radial_width)
        self.all_layers = [
            self.conv1,
            self.batch_normalization,
            self.softmax,
            self.sine_position_encoding,
            self.dense,
            self.sum,
            self.shifting_layer,
        ]

    def call(self, input_layer):
        self.input_layer = input_layer
        conv_layer = self.conv1(input_layer)
        conv_layer = self.batch_normalization(conv_layer)
        conv_layer = self.softmax(conv_layer)
        pos = self.sine_position_encoding(conv_layer)
        dense = self.dense(conv_layer + pos)
        dense = dense * 2 - 1
        shift_weights = self.sum(dense)
        output_layer = self.shifting_layer([input_layer, shift_weights])
        return output_layer

    def get_layer(self, name=None, index=None):
        if name:
            raise NotImplementedError()
        return self.all_layers[index]
