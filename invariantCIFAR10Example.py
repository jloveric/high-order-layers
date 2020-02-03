import tensorflow as tf
import snovalleyai_piecewise_polynomial_layers.PolynomialLayers as poly
from tensorflow.keras.layers import *
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = (x_train / 128.0 - 1.0), (x_test / 128.0 - 1.0)

units = 60
basis = poly.b2C
'''
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
'''

#Residual layer
def res_block(input_data, units=units, basis=basis) :
    x0 = LayerNormalization()(input_data)
    x1 = poly.Polynomial(units, basis=basis)(x0)
    x1 = Add()([x1, input_data])
    return x1

inputs = tf.keras.Input(shape=(32,32,3))
x = Flatten(input_shape=(32, 32, 3))(inputs)
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
