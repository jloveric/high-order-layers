from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import math

'''
Fourier series layer takes the number of frequencies desired
The total number of weights is 2*frequencies+1.  The first component
is the offset, and then sin and cos for each frequency.
TODO: add a per unit phase shift
'''
class Fourier(layers.Layer):

    '''
    units is the number of units in the layer
    input_dim is the number of dimensions in the input
    basis is an instance of the "Function" class which contains a basis and the number of weights
    TODO: almost identical to PolynomialLayers... so... reuse
    '''

    def __init__(self, units=None, frequencies=None, shift=0.0, length=2.0):
        super(Fourier, self).__init__()

        if units is None:
            print('You must define the units')
            raise

        if frequencies is None:
            print('You must define the frequencies.')
            raise

        self.units = units
        self.frequencies = frequencies
        self.numWeights = 2*frequencies+1
        self.length = length
        self.shift = shift

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(
                shape=(
                    self.units,
                    input_dim,
                    self.numWeights),
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

    def basisFourier(self, x, numFrequencies) :
        series = tf.convert_to_tensor([0*x+0.5-self.shift])
        for i in range(1,1+numFrequencies) :
            other = tf.convert_to_tensor([tf.math.cos(math.pi*i*x/self.length),tf.math.sin(math.pi*i*x/self.length)])
            series = tf.concat([series,other],axis=0)
        
        return series
    
    def call(self, inputs):

        shapeIn = inputs.shape
        shape = self.w.shape
        res = self.basisFourier(inputs,self.frequencies)
        res = tf.transpose(res,[1,2,0])
        res = tf.reshape(res, [-1, res.shape[1] * res.shape[2]])
        temp = tf.reshape(self.w, [-1, shape[1] * shape[2]])

        ans = tf.matmul(res, temp, transpose_a=False,
                        transpose_b=True)

        return ans
