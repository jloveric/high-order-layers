import tensorflow as tf
import snovalleyai_piecewise_polynomial_layers.PolynomialLayers as poly
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = (x_train / 128.0-1.0), (x_test / 128.0-1.0)

units = 20

basis = poly.b3

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  poly.Polynomial(units, 28*28, basis=basis),
  tf.keras.layers.LayerNormalization(),
  poly.Polynomial(units, units, basis=basis),
  tf.keras.layers.LayerNormalization(),
  poly.Polynomial(units, units, basis=basis),
  tf.keras.layers.LayerNormalization(),
  poly.Polynomial(units, units, basis=basis),
  tf.keras.layers.LayerNormalization(),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Batch norm should be messed up for batch_size=1, but seems to be working - perhaps not normalizing
model.fit(x_train, y_train, epochs=20, batch_size=10)
model.evaluate(x_test, y_test)