from typing import Tuple

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
    def __init__(self, width=0.25):
        super(ShiftingLayer, self).__init__()
        self.width = width
        self.amplitude = width**(1/2) * 1.772637204826652153

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        input_data, shifting_weights = inputs
        index_input = tf.reshape(tf.range(1, 1+tf.shape(input_data)[-2], dtype=shifting_weights.dtype), (1, 1, -1))
        index_shift = tf.reshape(tf.range(1, 1+shifting_weights.shape[-1], dtype=shifting_weights.dtype), (1, -1))
        index_shift += shifting_weights
        scaling_factor = tf.divide(tf.shape(input_data)[-2], shifting_weights.shape[-1])
        index_shift *= tf.cast(scaling_factor, dtype=index_shift.dtype)
        index_shift = tf.expand_dims(index_shift, axis=-1)
        M = index_input - index_shift
        M = tf.exp(-tf.square(M)/self.width)/self.amplitude
        return tf.matmul(M, input_data)


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


shifting_weights = np.array([[0]*90])
input_ = (np.ones(90)).reshape(1, -1, 1)
input1 = keras.layers.Input((None, 1))
shifting_input = keras.layers.Input((90))
sl = ShiftingLayer()([input1, shifting_input])
model = keras.models.Model(inputs=[input1, shifting_input], outputs=sl)
model.compile( run_eagerly=True)
model.summary()
y = model.predict([input_, shifting_weights])


plt.plot(input_[0,:])
# plt.plot(input_[1,:])
# plt.show()
#
plt.plot(y[0,:])
# plt.plot(y[1,:])
plt.show()