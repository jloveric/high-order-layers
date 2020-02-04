from . HighOrderLayer import *
from . Functions import *
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
Piecewise Lagrange polynomials with gauss lobatto points for tensorflow!
TODO: might move the polynomial derivation inside so the user does not need
to specify a basis, only the order and number of knots.
'''
class Polynomial(HighOrderLayer):
    def __init__(self, units=None, basis=None, shift=0.0,dtype=None):
        '''
        Parameters:
            units - is the number of units in the layer.
            basis - is an instance of the "Function" class which contains a basis and the number of weights.
            shift - shift the polynomial center by the given value.
            dtype - the data type for the weight arrays
        '''
        super(Polynomial, self).__init__(units=units,dtype=dtype)

        if units is None:
            print('You must define the units')
            raise

        if basis is None:
            print('You must define the basis.')
            raise

        self.units = units
        self.basis = basis
        self.shift = shift
        self.numWeights = self.basis.numWeights

    def function(self, inputs) :
        return self.basis(inputs-self.shift)
