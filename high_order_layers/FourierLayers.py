from .HighOrderLayer import *
import tensorflow as tf
import numpy as np
import math

'''
Fourier series layer takes the number of frequencies desired
The total number of weights is 2*frequencies+1.  The first component
is the offset, and then sin and cos for each frequency.
'''
class Fourier(HighOrderLayer):

    def __init__(self, units=None, frequencies=None, shift=0.0, length=2.0, dtype=None):
        '''
        Parameters :
            units - is the number of units in the layer
            frequencies - is the number of frequencies to include in fourier series
            shift - is an added phase shift
            length - is the wave length of the longest wave
            dtype - the data type for the tensorflow arrays
        '''
        super(Fourier, self).__init__(units=units,dtype=dtype)

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

    def function(self, inputs) :
        return self.basisFourier(inputs, self.frequencies)

    def basisFourier(self, x, numFrequencies) :
        series = tf.convert_to_tensor([0*x+0.5-self.shift])
        for i in range(1,1+numFrequencies) :
            other = tf.convert_to_tensor([tf.math.cos(math.pi*i*x/self.length),tf.math.sin(math.pi*i*x/self.length)])
            series = tf.concat([series,other],axis=0)
        
        return series
