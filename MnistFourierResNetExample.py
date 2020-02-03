#Example using fourier series layer in a residual network.  Not particularly great
#on this problem, but it may do better on time series.

import tensorflow as tf
from snovalleyai_piecewise_polynomial_layers import FourierLayers as fourier
from tensorflow.keras.layers import *
mnist = tf.keras.datasets.mnist
layers = tf.keras.layers

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = (x_train / 128.0 - 1.0), (x_test / 128.0 - 1.0)

units = 5
frequencies = 10

#Residual layer
def res_block(input_data, units=units, frequencies=frequencies) :
    x0 = LayerNormalization()(input_data)
    x1 = fourier.Fourier(units, frequencies=frequencies)(x0)
    x1 = Add()([x1, input_data])
    return x1

inputs = tf.keras.Input(shape=(28,28))
x = Flatten(input_shape=(28, 28))(inputs)
x = fourier.Fourier(units, frequencies=frequencies)(x)
for i in range(3) :
    x = res_block(x, frequencies=frequencies, units=units)
x = LayerNormalization()(x)
outputs = Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=10)
model.evaluate(x_test, y_test)
