import tensorflow as tf
import high_order_layers.PolynomialLayers as poly
import high_order_layers.HighOrderConvolution2D as pconv
from tensorflow.keras import datasets, models
from tensorflow.keras.layers import *
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

mnist = tf.keras.datasets.mnist

(x_left, y_left), (x_test, y_test) = mnist.load_data()
x_left, x_test = x_left / 255.0, x_test / 255.0
x_left = x_left.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

#Also create a training and validation set
x_train, x_valid, y_train, y_valid = train_test_split(x_left, y_left, test_size=6000)

units = 20

basis = poly.b3

inputs = tf.keras.Input(shape=(28,28,1))
x = pconv.high_order_convolution2D(inputs,8,(3,3),basis=basis)
x = MaxPooling2D((2, 2))(x)
x = pconv.high_order_convolution2D(x,16,(3,3),basis=basis)
x = MaxPooling2D((2, 2))(x)
x = pconv.high_order_convolution2D(x,16,(3,3),basis=basis)
x = Flatten()(x)
x = poly.Polynomial(units, basis=basis)(x)
x = LayerNormalization()(x)
x = poly.Polynomial(units, basis=basis)(x)
x = LayerNormalization()(x)
outputs = Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=False)

model.fit(x_train, y_train, epochs=20, batch_size=10, validation_data=(x_valid, y_valid))
model.evaluate(x_test, y_test)
