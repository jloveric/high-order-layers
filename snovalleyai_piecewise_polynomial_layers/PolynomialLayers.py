'''
Piecewise Lagrange polynomials with gauss lobatto points for tensorflow!
'''
from . Functions import *
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

"""
Store the function and the number of weights in the
same object.
"""

class FunctionWrapper():
    def __init__(self, basis, numWeights):
        self.numWeights = numWeights
        self.basis = basis

    def __call__(self, val):
        return self.basis(val)

b0 = FunctionWrapper(basis0, 1)

b1 = FunctionWrapper(basis1, 2)
b1D = FunctionWrapper(basis1DG, 4)
b1C = FunctionWrapper(basis1CG, 3)

b2 = FunctionWrapper(basis2, 3)
b2C = FunctionWrapper(basis2CG, 5)
b2D = FunctionWrapper(basis2DG, 6)

b3 = FunctionWrapper(basis3, 4)
b3C = FunctionWrapper(basis3CG, 7)
b3D = FunctionWrapper(basis3DG, 8)

b4 = FunctionWrapper(basis4, 5)
b4C = FunctionWrapper(basis4CG, 9)
b4D = FunctionWrapper(basis4DG, 10)

b5 = FunctionWrapper(basis5, 6)
b5C = FunctionWrapper(basis5CG, 11)
b5D = FunctionWrapper(basis5DG, 12)

'''
Tensorflow layer that takes a function as a parameter
'''
class Polynomial(layers.Layer):

    '''
    units is the number of units in the layer
    input_dim is the number of dimensions in the input
    basis is an instance of the "Function" class which contains a basis and the number of weights
    '''

    def __init__(self, units=None, basis=None):
        super(Polynomial, self).__init__()

        if units is None:
            print('You must define the units')
            raise

        if basis is None:
            print('You must define the basis.')
            raise

        self.units = units
        self.basis = basis

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(
                shape=(
                    self.units,
                    input_dim,
                    self.basis.numWeights),
                dtype='float32'),
            trainable=True)

        # Set all these to zero
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(
                shape=(
                    self.units,
                ),
                dtype='float32'),
            trainable=True)

    def call(self, inputs):

        shapeIn = inputs.shape
        shape = self.w.shape

        print('shape', shape, 'shapeIn', shapeIn)

        res = self.basis(inputs)
        
        print('res.shape', res.shape)

        res = tf.transpose(res,[2,1,0])
        res = tf.reshape(res, [-1, res.shape[0] * res.shape[2]])
        temp = tf.reshape(self.w, [-1, shape[1] * shape[2]])

        ans = tf.matmul(res, temp, transpose_a=False,
                        transpose_b=True)

        return ans
