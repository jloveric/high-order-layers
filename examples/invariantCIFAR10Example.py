import sys
sys.path.append('../')

import tensorflow as tf
import snovalleyai_piecewise_polynomial_layers.PolynomialLayers as poly
from tensorflow.keras.layers import *
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train),(x_test, y_test) = cifar10.load_data()
x_train, x_test = (x_train / 128.0-1.0), (x_test / 128.0-1.0)

units = 60

basis = poly.b2C

model = tf.keras.models.Sequential([
  Flatten(input_shape=(32, 32, 3)),
  poly.Polynomial(units, basis=basis),
  LayerNormalization(),
  poly.Polynomial(units, basis=basis),
  LayerNormalization(),
  poly.Polynomial(units, basis=basis),
  LayerNormalization(),
  poly.Polynomial(units, basis=basis),
  LayerNormalization(),
  Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=10)
model.evaluate(x_test, y_test)