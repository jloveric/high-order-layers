import tensorflow as tf
import PolynomialLayers as poly
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = (x_train / 128.0-1.0), (x_test / 128.0-1.0)

units = 128

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  poly.Polynomial(units, 28*28),
  #tf.keras.layers.ReLU(512),
  tf.keras.layers.LayerNormalization(),
  tf.keras.layers.Dropout(0.2),
  poly.Polynomial(units, units),
  #tf.keras.layers.ReLU(512),
  tf.keras.layers.LayerNormalization(),
  tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Dense(512, activation='relu'),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])


'''
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  #poly.Polynomial(512, 28*28),
  #tf.keras.layers.ReLU(512),
  tf.keras.layers.Dense(512, activation='relu'),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
'''

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)