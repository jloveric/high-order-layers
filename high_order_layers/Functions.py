import tensorflow as tf
import numpy as np
import sys
import math

# Polynomial basis functions
def basis0(x):
    return tf.convert_to_tensor([x])

def basis1(x):
    return tf.convert_to_tensor([-0.5 * (x - 1.0), 0.5 * (x + 1.0)])

# quadratic polynomial
def basis2(xIn):
    x = tf.convert_to_tensor(xIn)
    return tf.convert_to_tensor(
        [0.5 * x * (x - 1), -1.0 * (x + 1) * (x - 1), 0.5 * x * (x + 1)])

# cubic polynomial
def basis3(xIn):
    x = xIn
    eta = x
    powEta2 = eta * x
    powEta3 = powEta2 * x

    temp = tf.convert_to_tensor([-0.6666666666666666 * powEta3 + 0.6666666666666666 * powEta2 +
                                 0.1666666666666667 * eta - 0.1666666666666667,
                                 1.333333333333333 * powEta3 - 0.6666666666666666 * powEta2 -
                                 1.333333333333333 * eta + 0.6666666666666666,
                                 -1.333333333333333 * powEta3 - 0.6666666666666666 * powEta2 +
                                 1.333333333333333 * eta + 0.6666666666666666,
                                 0.6666666666666666 * powEta3 + 0.6666666666666666 * powEta2 -
                                 0.1666666666666667 * eta - 0.1666666666666667])
    return temp

# quartic polynomial
def basis4(xIn):
    x = xIn

    eta = x
    powEta2 = x * x
    powEta3 = powEta2 * x
    powEta4 = powEta3 * x

    temp = tf.convert_to_tensor([powEta4 -
                                 powEta3 -
                                 0.5 *
                                 powEta2 +
                                 0.5 *
                                 eta, -
                                 2.0 *
                                 powEta4 +
                                 1.414213562373095 *
                                 powEta3 +
                                 2.0 *
                                 powEta2 -
                                 1.414213562373095 *
                                 eta, 1 -
                                 3 *
                                 powEta2 +
                                 2 *
                                 powEta4, -
                                 2 *
                                 powEta4 -
                                 1.414213562373095 *
                                 powEta3 +
                                 2.0 *
                                 powEta2 +
                                 1.414213562373095 *
                                 eta, powEta4 +
                                 powEta3 -
                                 0.5 *
                                 powEta2 -
                                 0.5 *
                                 eta])

    return temp

# quintic polynomial


def basis5(xIn):
    #x = np.tanh(xIn)
    x = xIn
    eta = x
    powEta2 = eta * eta
    powEta3 = eta * powEta2
    powEta4 = eta * powEta3
    powEta5 = eta * powEta4

    temp = tf.convert_to_tensor([(0.1 -
                                  0.1 *
                                  eta -
                                  1.2 *
                                  powEta2 +
                                  1.2 *
                                  powEta3 +
                                  1.6 *
                                  powEta4 -
                                  1.6 *
                                  powEta5), 3.2 *
                                 powEta5 -
                                 2.588854381999832 *
                                 powEta4 -
                                 3.505572809000084 *
                                 powEta3 +
                                 2.83606797749979 *
                                 powEta2 +
                                 0.3055728090000842 *
                                 eta -
                                 0.247213595499958, -
                                 3.2 *
                                 powEta5 +
                                 0.9888543819998319 *
                                 powEta4 +
                                 5.294427190999916 *
                                 powEta3 -
                                 1.63606797749979 *
                                 powEta2 -
                                 2.094427190999916 *
                                 eta +
                                 0.6472135954999582, 3.2 *
                                 powEta5 +
                                 0.9888543819998319 *
                                 powEta4 -
                                 5.294427190999916 *
                                 powEta3 -
                                 1.63606797749979 *
                                 powEta2 +
                                 2.094427190999916 *
                                 eta +
                                 0.6472135954999582, -
                                 3.2 *
                                 powEta5 -
                                 2.588854381999832 *
                                 powEta4 +
                                 3.505572809000084 *
                                 powEta3 +
                                 2.83606797749979 *
                                 powEta2 -
                                 0.3055728090000842 *
                                 eta -
                                 0.247213595499958, (0.1 +
                                                     0.1 *
                                                     eta -
                                                     1.2 *
                                                     powEta2 -
                                                     1.2 *
                                                     powEta3 +
                                                     1.6 *
                                                     powEta4 +
                                                     1.6 *
                                                     powEta5)])

    return temp

'''
Discontinuous polynomial, 2 parts
'''
def basisDG(x, basis) :
    xr = tf.where(x > 0, 2.0 * (x - 0.5), 0 * x)
    xl = tf.where(x <= 0, 2.0 * (x + 0.5), 0 * x)
    pos = tf.where(x > 0, 1.0, 0.0)
    neg = tf.where(x <= 0, 1.0, 0.0)

    fr = pos*basis(xr)
    fl = neg*basis(xl)
    f = tf.concat([fl,fr],axis=0)
    
    return f

def basis1DG(x):
    return basisDG(x, basis1)

def basis2DG(x):
    return basisDG(x, basis2)

def basis3DG(x):
    return basisDG(x, basis3)

def basis4DG(x):
    return basisDG(x, basis4)

def basis5DG(x):
    return basisDG(x, basis5)

'''
Continuous piecewise polynomials
'''
def basisCG(x, basis, width) :
    #Add one additional dimension for the batch, I believe.
    dims = len(x.get_shape().as_list())+1
    
    lR = [ [0]*2 for _ in range(dims) ]
    lL = [ [0]*2 for _ in range(dims) ]

    lR[0][0] = width-1
    lL[0][1] = width-1

    paddingsR = tf.convert_to_tensor(lR)
    paddingsL = tf.convert_to_tensor(lL)

    xr = tf.where(x > 0, 2.0 * (x - 0.5), 0 * x)
    xl = tf.where(x <= 0, 2.0 * (x + 0.5), 0 * x)
    pos = tf.where(x > 0, 1.0, 0.0)
    neg = tf.where(x <= 0, 1.0, 0.0)

    fr = tf.pad(pos*basis(xr),paddingsR)
    fl = tf.pad(neg*basis(xl),paddingsL)
    
    f = fr+fl
    
    return f

def basis1CG(x):
    return basisCG(x, basis1, 2)

def basis2CG(x):
    return basisCG(x, basis2, 3)

def basis3CG(x):
    return basisCG(x, basis3, 4)

def basis4CG(x):
    return basisCG(x, basis4, 5)

def basis5CG(x):
    return basisCG(x, basis5, 6)
    
