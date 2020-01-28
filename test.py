import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3" 

import unittest
from snovalleyai_piecewise_polynomial_layers.PolynomialLayers import *

from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np

class TestPolynomials(unittest.TestCase):

    def test_basis(self):
        # Basic test that there is no crash.
        # TODO: port the c++ tests over
        b0(tf.convert_to_tensor([[1.0]]))

        b1(tf.convert_to_tensor([[1.0,2.0]]))
        b1D(tf.convert_to_tensor([[1.0]]))
        b1C(tf.convert_to_tensor([[1.0]]))

        b2(tf.convert_to_tensor([[1.0]]))
        b2C(tf.convert_to_tensor([[1.0]]))
        b2D(tf.convert_to_tensor([[1.0]]))

        b3(tf.convert_to_tensor([[1.0]]))
        b4(tf.convert_to_tensor([[1.0]]))

        b5(tf.convert_to_tensor([[1.0]]))
        b5C(tf.convert_to_tensor([[1.0]]))
        b5D(tf.convert_to_tensor([[1.0]]))
        print('finished basis test')

    #Integration test for single input
    def test_single_input(self) :
        try :
            offset = -0.1
            factor = 1.5 * 3.14159
            xTest = np.arange(100) / 50 - 1.0
            yTest = 0.5 * np.cos(factor * (xTest - offset))

            xTrain = tf.random.uniform([1000], minval=-1.0, maxval=1, dtype=tf.float32)
            yTrain = 0.5 * tf.math.cos(factor * (xTrain - offset))

            basis = b5
            model = tf.keras.models.Sequential([
                Polynomial(1, basis=basis),
            ])

            model.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['accuracy'])

            model.fit(xTrain, yTrain, epochs=1, batch_size=100)
            model.evaluate(xTrain, yTrain)
            self.assertTrue(True)
        except :
            print('Single input example failed.')
            self.assertTrue(False) 
        
    #Integration test for multiple input
    def test_multi_input(self) :
        
        try :
            mnist = tf.keras.datasets.mnist
            layers = tf.keras.layers

            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, x_test = (x_train / 128.0 - 1.0), (x_test / 128.0 - 1.0)

            units = 1

            basis = b5D

            model = tf.keras.models.Sequential([
                Flatten(input_shape=(28, 28)),
                Polynomial(units, basis=basis),
                LayerNormalization(),
                Dense(10, activation='softmax')
            ])

            model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

            model.fit(x_train, y_train, epochs=1, batch_size=10000)
            model.evaluate(x_test, y_test)
            print('finished integration test')
            self.assertTrue(True)
        except :
            print('invariant mnist example crashed.')
            self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()