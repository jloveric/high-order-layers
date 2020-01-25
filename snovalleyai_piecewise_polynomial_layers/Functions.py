import tensorflow as tf
import numpy as np
import sys


def swapAxes(x):
    return tf.transpose(x, [1, 2, 0])

def basis0(x):
    return tf.convert_to_tensor([x])

def basis1(x):
    return tf.convert_to_tensor([-0.5 * (x - 1.0), 0.5 * (x + 1.0)])
    #return temp #swapAxes(temp)

# quadratic polynomial
def basis2(xIn):
    # flatten
    x = tf.convert_to_tensor(xIn)
    #x = xIn
    temp = tf.convert_to_tensor(
        [0.5 * x * (x - 1), -1.0 * (x + 1) * (x - 1), 0.5 * x * (x + 1)])

    return temp #tf.transpose(temp, [1, 2, 0])
    # return swapAxes(temp)

# cubic polynomial
def basis3(xIn):
    x = xIn  # np.tanh(xIn)
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
    x = xIn  # np.tanh(xIn)

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

    return temp #swapAxes(temp)

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
TODO: figure out how to combine this like above.  It's a little more difficult
since there is overlap which alternates between left and right.
'''
def basis1CG(x):
    xr = tf.where(x > 0, 2.0 * (x - 0.5), 0 * x)
    xl = tf.where(x <= 0, 2.0 * (x + 0.5), 0 * x)
    pos = tf.where(x > 0, 1.0, 0.0)
    neg = tf.where(x <= 0, 1.0, 0.0)

    res1 = pos * \
        tf.convert_to_tensor([xr * 0.0 + 1e-6, -0.5 * (xr - 1.0), 0.5 * (xr + 1.0)])
    res2 = neg * \
        tf.convert_to_tensor([-0.5 * (xl - 1.0), 0.5 * (xl + 1.0), xl * 0.0 + 1e-6])

    return res1 + res2


def basis2CG(x):
    xr = tf.where(x > 0, 2.0 * (x - 0.5), 0 * x)
    xl = tf.where(x <= 0, 2.0 * (x + 0.5), 0 * x)
    pos = tf.where(x > 0, 1.0, 0.0)
    neg = tf.where(x <= 0, 1.0, 0.0)

    res1 = pos * tf.convert_to_tensor([x * 0.0, x * 0.0, 0.5 * xr * (
        xr - 1.0), -1.0 * (xr + 1) * (xr - 1), 0.5 * xr * (xr + 1.0)])
    res2 = neg * tf.convert_to_tensor([0.5 * xl * (xl - 1.0), -1.0 * (
        xl + 1) * (xl - 1), 0.5 * xl * (xl + 1.0), x * 0.0, x * 0.0])

    return res1 + res2

def basis5CG(x):

    pos = tf.where(x > 0, 1.0, 0.0)
    neg = tf.where(x <= 0, 1.0, 0.0)

    xr = tf.where(x > 0, 2.0 * (x - 0.5), 0 * x)
    xl = tf.where(x <= 0, 2.0 * (x + 0.5), 0 * x)

    eta = xr
    powEta2 = eta * eta
    powEta3 = eta * powEta2
    powEta4 = eta * powEta3
    powEta5 = eta * powEta4

    res1 = pos * tf.convert_to_tensor([eta * 0, eta * 0, eta * 0, eta * 0, eta * 0, (0.1 - 0.1 * eta - 1.2 * powEta2 + 1.2 * powEta3 + 1.6 * powEta4 - 1.6 * powEta5),
                                       3.2 * powEta5 - 2.588854381999832 * powEta4 - 3.505572809000084 * powEta3 +
                                       2.83606797749979 * powEta2 + 0.3055728090000842 * eta - 0.247213595499958,
                                       -3.2 * powEta5 + 0.9888543819998319 * powEta4 + 5.294427190999916 * powEta3 -
                                       1.63606797749979 * powEta2 - 2.094427190999916 * eta + 0.6472135954999582,
                                       3.2 * powEta5 + 0.9888543819998319 * powEta4 - 5.294427190999916 * powEta3 -
                                       1.63606797749979 * powEta2 + 2.094427190999916 * eta + 0.6472135954999582,
                                       -3.2 * powEta5 - 2.588854381999832 * powEta4 + 3.505572809000084 * powEta3 +
                                       2.83606797749979 * powEta2 - 0.3055728090000842 * eta - 0.247213595499958,
                                       (0.1 + 0.1 * eta - 1.2 * powEta2 - 1.2 * powEta3 + 1.6 * powEta4 + 1.6 * powEta5)])

    eta = xl
    powEta2 = eta * eta
    powEta3 = eta * powEta2
    powEta4 = eta * powEta3
    powEta5 = eta * powEta4

    res2 = neg * tf.convert_to_tensor([(0.1 - 0.1 * eta - 1.2 * powEta2 + 1.2 * powEta3 + 1.6 * powEta4 - 1.6 * powEta5),
                                       3.2 * powEta5 - 2.588854381999832 * powEta4 - 3.505572809000084 * powEta3 +
                                       2.83606797749979 * powEta2 + 0.3055728090000842 * eta - 0.247213595499958,
                                       -3.2 * powEta5 + 0.9888543819998319 * powEta4 + 5.294427190999916 * powEta3 -
                                       1.63606797749979 * powEta2 - 2.094427190999916 * eta + 0.6472135954999582,
                                       3.2 * powEta5 + 0.9888543819998319 * powEta4 - 5.294427190999916 * powEta3 -
                                       1.63606797749979 * powEta2 + 2.094427190999916 * eta + 0.6472135954999582,
                                       -3.2 * powEta5 - 2.588854381999832 * powEta4 + 3.505572809000084 * powEta3 +
                                       2.83606797749979 * powEta2 - 0.3055728090000842 * eta - 0.247213595499958,
                                       (0.1 + 0.1 * eta - 1.2 * powEta2 - 1.2 * powEta3 + 1.6 * powEta4 + 1.6 * powEta5), eta * 0, eta * 0, eta * 0, eta * 0, eta * 0])

    return res1 + res2
