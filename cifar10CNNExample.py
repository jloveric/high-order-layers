import tensorflow as tf
import snovalleyai_piecewise_polynomial_layers.PolynomialLayers as poly
from tensorflow.keras import datasets, layers, models

cifar10 = tf.keras.datasets.cifar10

(x_train, y_train),(x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train/255.0, x_test / 255.0

units = 20

basis = poly.b3

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(poly.Polynomial(units, basis=basis))
model.add(tf.keras.layers.LayerNormalization())
model.add(poly.Polynomial(units, basis=basis))
model.add(tf.keras.layers.LayerNormalization())
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=10)
model.evaluate(x_test, y_test)