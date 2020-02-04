#Example using polynomial layer in a residual network.  Not particularly great
#on this problem, but it may do better on time series.

import tensorflow as tf
from high_order_layers import PolynomialLayers as poly
from tensorflow.keras.layers import *
mnist = tf.keras.datasets.mnist
layers = tf.keras.layers

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = (x_train / 128.0 - 1.0), (x_test / 128.0 - 1.0)

units = 10
basis = poly.b3

#Residual layer
def res_block(input_data, units=units, basis=basis) :
    x0 = LayerNormalization()(input_data)
    x1 = poly.Polynomial(units, basis=basis)(x0)
    x1 = Add()([x1, input_data])
    return x1

inputs = tf.keras.Input(shape=(28,28))
x = Flatten(input_shape=(28, 28))(inputs)
x = poly.Polynomial(units, basis=basis)(x)
for i in range(3) :
    x = res_block(x, basis=basis, units=units)
x = LayerNormalization()(x)
outputs = Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=10)
model.evaluate(x_test, y_test)
