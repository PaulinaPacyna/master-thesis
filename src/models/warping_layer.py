import keras_nlp
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ReductionLayer(keras.layers.Layer):
    def __init__(self, axis=1):
        super(ReductionLayer, self).__init__()
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)


class ShiftingLayer(keras.layers.Layer):
    def __init__(self, target_length, width=0.25):
        super(ShiftingLayer, self).__init__()
        self.width = width
        self.target_length = target_length
    def build(self, input_series_shape):
        print(input_series_shape)
        a = np.arange(input_series_shape[-2])
        b = np.arange(self.target_length).reshape((-1, 1))
        shifting_weight_matrix = a - b
        self.input_series_shape = input_series_shape
        self.shifting_weight_matrix = tf.Variable(shifting_weight_matrix, trainable=False)
        print(self.shifting_weight_matrix)
        self.scaling_factor = tf.Variable(input_series_shape[1] / self.target_length,
                                          trainable=False)  # TODO: incorporate

    def call(self, input, shifting_weights):
        shifting_weights = tf.repeat(tf.expand_dims(shifting_weights, axis=-1), repeats=self.input_series_shape[-2], axis=-1)
        shifting_weights = shifting_weights + self.shifting_weight_matrix
        shifting_weights = tf.exp(-tf.square(shifting_weights) / self.width)
        return tf.matmul(shifting_weights, input)


def warping_layer(output_length, input_lenght=None, number_of_convolutions=64, kernel_size=11) -> keras.models.Model:
    input_layer = keras.layers.Input((input_lenght, 1))
    conv_layer = keras.layers.Conv1D(filters=number_of_convolutions, kernel_size=kernel_size, padding="same")(
        input_layer
    )
    conv_layer = keras.layers.BatchNormalization()(conv_layer)
    conv_layer = keras.layers.Softmax(axis=1)(conv_layer)
    pos = keras_nlp.layers.SinePositionEncoding()(conv_layer)
    dense = keras.layers.Dense(units=output_length, activation='sigmoid')(conv_layer + pos)
    dense = dense * 2 - 1
    shift_weights = ReductionLayer(axis=1)(dense)
    output_layer = 1
    model = keras.models.Model(inputs=input_layer, outputs=shift_weights)


shifting_weights = tf.Variable(np.array([[0, -1, -2, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1, 1, 1]]))
input_ = tf.Variable(np.cos(np.arange(20)).reshape(2, 10))
plt.plot(input_.numpy()[0,:])
plt.plot(input_.numpy()[1,:])
plt.show()

input1 = keras.layers.Input((8, 1))
shifting_input = keras.layers.Input((10,))
sl = ShiftingLayer(10)(input1, shifting_input)