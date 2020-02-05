import tensorflow as tf
import high_order_layers.PolynomialLayers as poly
import high_order_layers.HighOrderConvolution2D as pconv
from tensorflow.keras import datasets, models
from tensorflow.keras.layers import *

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

units = 20

basis = poly.b3

inputs = tf.keras.Input(shape=(28,28,1))
x = pconv.high_order_convolution2D(inputs,8,(3,3),basis=basis)
x = MaxPooling2D((2, 2))(x)
x = pconv.high_order_convolution2D(x,8,(3,3),basis=basis)
x = MaxPooling2D((2, 2))(x)
x = pconv.high_order_convolution2D(x,8,(3,3),basis=basis)
x=Flatten()(x)
x=poly.Polynomial(units, basis=basis)(x)
x=LayerNormalization()(x)
x=poly.Polynomial(units, basis=basis)(x)
x=LayerNormalization()(x)
outputs = Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(x_train, y_train, epochs=20, batch_size=10, validation_data=(x_train, y_train))
model.evaluate(x_test, y_test)
