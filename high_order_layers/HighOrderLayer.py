from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

'''
Base class for high order layers.
'''
class HighOrderLayer(layers.Layer):

    '''
    units is the number of units in the layer
    input_dim is the number of dimensions in the input
    '''
    def __init__(self, units=None, dtype=None):
        super(HighOrderLayer, self).__init__()
        self.numWeights = None
        self.units = None

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(
                shape=(
                    self.units,
                    input_dim,
                    self.numWeights),
                dtype=self.dtype),
            trainable=True)

        # We don't need b
        '''
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(
                shape=(
                    self.units,
                ),
                dtype=self.dtype),
            trainable=True)
        '''

    def function(self, x) :
        raise Exception('Function not implemented in '+type(self).__name__)
       
    def call(self, inputs):

        shapeIn = inputs.shape
        shape = self.w.shape
        res = self.function(inputs)
        res = tf.transpose(res,[1,2,0])
        res = tf.reshape(res, [-1, res.shape[1] * res.shape[2]])
        temp = tf.reshape(self.w, [-1, shape[1] * shape[2]])

        ans = tf.matmul(res, temp, transpose_a=False,
                        transpose_b=True)

        return ans
