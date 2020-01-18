import tensorflow as tf
import numpy as np


def swapAxes(x) :
    return tf.transpose(x,[1,2,0])
    #return np.swapaxes(np.swapaxes(x,0,1),1,2)

# constant
def basis0(x) :
    return swapAxes(tf.convert_to_tensor([x]))

# linear (with offset)
def basis1(x) :
    temp =  tf.convert_to_tensor([-0.5*(x-1.0),0.5*(x+1.0)])
    return swapAxes(temp)

# piecewice discontinuous linear
def basis1DG(xIn) :
    return basisSplit(xIn, basis1DG1)

#quadratic polynomial
def basis2(xIn) :
    #flatten
    x = tf.convert_to_tensor(xIn)
    #x = xIn
    temp = tf.convert_to_tensor([0.5*x*(x-1), -1.0*(x+1)*(x-1), 0.5*x*(x+1)])
    
    return tf.transpose(temp,[1,2,0])
    #return swapAxes(temp)

#piecewise continuous quadratic
def basis2CG(xIn) :
    return basisSplit(xIn, basis2CG1)

#piecewise discontinuous quadratic
def basis2DG(xIn) :
    return basisSplit(xIn, basis2DG1)

#cubic polynomial
def basis3(xIn) :
    x = xIn #np.tanh(xIn)
    eta = x
    powEta2 = eta*x
    powEta3 = powEta2*x

    temp = tf.convert_to_tensor([-0.6666666666666666 * powEta3 + 0.6666666666666666 * powEta2 +
            0.1666666666666667 * eta - 0.1666666666666667,
   1.333333333333333 * powEta3 - 0.6666666666666666 * powEta2 -
            1.333333333333333 * eta + 0.6666666666666666,
   -1.333333333333333 * powEta3 - 0.6666666666666666 * powEta2 +
            1.333333333333333 * eta + 0.6666666666666666,
   0.6666666666666666 * powEta3 + 0.6666666666666666 * powEta2 -
            0.1666666666666667 * eta - 0.1666666666666667])
    return swapAxes(temp)

# quartic polynomial
def basis4(xIn) :
    x = xIn #np.tanh(xIn)
    
    eta = x
    powEta2 = x * x
    powEta3 = powEta2 * x
    powEta4 = powEta3 * x

    temp = tf.convert_to_tensor([powEta4 - powEta3 - 0.5 * powEta2 + 0.5 * eta,
        -2.0 * powEta4 + 1.414213562373095 * powEta3 + 2.0 * powEta2 -
        1.414213562373095 * eta,
        1 - 3 * powEta2 + 2 * powEta4,
        -2 * powEta4 - 1.414213562373095 * powEta3 + 2.0 * powEta2 +
        1.414213562373095 * eta,
        powEta4 + powEta3 - 0.5 * powEta2 - 0.5 * eta])
    
    return swapAxes(temp)

#quintic polynomial
def basis5(xIn) :
    #x = np.tanh(xIn)
    x = xIn
    eta = x
    powEta2 = eta * eta
    powEta3 = eta * powEta2
    powEta4 = eta * powEta3
    powEta5 = eta * powEta4

    temp = tf.convert_to_tensor([(0.1 - 0.1 * eta - 1.2 * powEta2 + 1.2 * powEta3 + 1.6 * powEta4 - 1.6 * powEta5),
        3.2 * powEta5 - 2.588854381999832 * powEta4 - 3.505572809000084 * powEta3 +
        2.83606797749979 * powEta2 + 0.3055728090000842 * eta - 0.247213595499958,
        -3.2 * powEta5 + 0.9888543819998319 * powEta4 + 5.294427190999916 * powEta3 -
        1.63606797749979 * powEta2 - 2.094427190999916 * eta + 0.6472135954999582,
        3.2 * powEta5 + 0.9888543819998319 * powEta4 - 5.294427190999916 * powEta3 -
        1.63606797749979 * powEta2 + 2.094427190999916 * eta + 0.6472135954999582,
        -3.2 * powEta5 - 2.588854381999832 * powEta4 + 3.505572809000084 * powEta3 +
        2.83606797749979 * powEta2 - 0.3055728090000842 * eta - 0.247213595499958,
        (0.1 + 0.1 * eta - 1.2 * powEta2 - 1.2 * powEta3 + 1.6 * powEta4 + 1.6 * powEta5)])

    return swapAxes(temp)

#5th order piecwise continuous polynomial
def basis5CG(xIn) :
    return basisSplit(xIn, basis5CG1)

#5th order piecwise discontinous polynomial
def basis5DG(xIn) :
    return basisSplit(xIn, basis5DG1)

'''
The functions below should not be called directly, instead call the wrappers above
'''
def basis1DG1(x) :
    xr = tf.where(x>0,2.0*(x-0.5),0*x)
    xl = tf.where(x<=0,2.0*(x+0.5),0*x)

    res1 = tf.convert_to_tensor([xr*0.0, xr*0.0, -0.5*(xr-1.0),0.5*(xr+1.0)])
    res2 = tf.convert_to_tensor([-0.5*(xl-1.0),0.5*(xl+1.0),xl*0.0,xl*0.0])

    return res1 + res2

def basis2CG1(x) :
    xr = tf.where(x>0,2.0*(x-0.5),0*x)
    xl = tf.where(x<=0,2.0*(x+0.5),0*x)

    res1 = tf.convert_to_tensor([x*0.0, x*0.0, 0.5*xr*(xr-1.0),-1.0*(xr+1)*(xr-1), 0.5*xr*(xr+1.0)])
    res2 = tf.convert_to_tensor([0.5*xl*(xl-1.0),-1.0*(xl+1)*(xl-1),0.5*xl*(xl+1.0),x*0.0,x*0.0])

    return res1+res2

def basis2DG1(x) :
    xr = tf.where(x>0,2.0*(x-0.5),0*x)
    xl = tf.where(x<=0,2.0*(x+0.5),0*x)
    
    res1 = tf.convert_to_tensor([x*0.0, x*0.0, x*0.0, 0.5*xr*(xr-1.0),-1.0*(xr+1)*(xr-1), 0.5*xr*(xr+1.0)])
    res2 = tf.convert_to_tensor([0.5*xl*(xl-1.0),-1.0*(xl+1)*(xl-1),0.5*xl*(xl+1.0),x*0.0,x*0.0,x*0.0])
    
    return res1 + res2

def basis5DG1(x) :
    xr = tf.where(x>0,2.0*(x-0.5),0*x)

    eta = xr
    powEta2 = eta * eta
    powEta3 = eta * powEta2
    powEta4 = eta * powEta3
    powEta5 = eta * powEta4
    res1 = tf.convert_to_tensor([0*eta, 0*eta, 0*eta, 0*eta, 0*eta, 0*eta, (0.1 - 0.1 * eta - 1.2 * powEta2 + 1.2 * powEta3 + 1.6 * powEta4 - 1.6 * powEta5),
                3.2 * powEta5 - 2.588854381999832 * powEta4 - 3.505572809000084 * powEta3 +
                2.83606797749979 * powEta2 + 0.3055728090000842 * eta - 0.247213595499958,
                -3.2 * powEta5 + 0.9888543819998319 * powEta4 + 5.294427190999916 * powEta3 -
                1.63606797749979 * powEta2 - 2.094427190999916 * eta + 0.6472135954999582,
                3.2 * powEta5 + 0.9888543819998319 * powEta4 - 5.294427190999916 * powEta3 -
                1.63606797749979 * powEta2 + 2.094427190999916 * eta + 0.6472135954999582,
                -3.2 * powEta5 - 2.588854381999832 * powEta4 + 3.505572809000084 * powEta3 +
                2.83606797749979 * powEta2 - 0.3055728090000842 * eta - 0.247213595499958,
                (0.1 + 0.1 * eta - 1.2 * powEta2 - 1.2 * powEta3 + 1.6 * powEta4 + 1.6 * powEta5)])
    
    #xl = 2.0*(x[x<=0]+0.5)
    xl = tf.where(x<=0,2.0*(x+0.5),0*x)
    
    eta = xl
    powEta2 = eta * eta
    powEta3 = eta * powEta2
    powEta4 = eta * powEta3
    powEta5 = eta * powEta4
    res2 = tf.convert_to_tensor([(0.1 - 0.1 * eta - 1.2 * powEta2 + 1.2 * powEta3 + 1.6 * powEta4 - 1.6 * powEta5),
                3.2 * powEta5 - 2.588854381999832 * powEta4 - 3.505572809000084 * powEta3 +
                2.83606797749979 * powEta2 + 0.3055728090000842 * eta - 0.247213595499958,
                -3.2 * powEta5 + 0.9888543819998319 * powEta4 + 5.294427190999916 * powEta3 -
                1.63606797749979 * powEta2 - 2.094427190999916 * eta + 0.6472135954999582,
                3.2 * powEta5 + 0.9888543819998319 * powEta4 - 5.294427190999916 * powEta3 -
                1.63606797749979 * powEta2 + 2.094427190999916 * eta + 0.6472135954999582,
                -3.2 * powEta5 - 2.588854381999832 * powEta4 + 3.505572809000084 * powEta3 +
                2.83606797749979 * powEta2 - 0.3055728090000842 * eta - 0.247213595499958,
                (0.1 + 0.1 * eta - 1.2 * powEta2 - 1.2 * powEta3 + 1.6 * powEta4 + 1.6 * powEta5),0*eta, 0*eta, 0*eta, 0*eta, 0*eta, 0*eta])
    
    return res1+res2

def basis5CG1(x) :

    xr = tf.where(x>0,2.0*(x-0.5),0*x)
    xl = tf.where(x<=0,2.0*(x+0.5),0*x)

    eta = xr
    powEta2 = eta * eta
    powEta3 = eta * powEta2
    powEta4 = eta * powEta3
    powEta5 = eta * powEta4

    res1 = tf.convert_to_tensor([eta*0, eta*0, eta*0, eta*0, eta*0, (0.1 - 0.1 * eta - 1.2 * powEta2 + 1.2 * powEta3 + 1.6 * powEta4 - 1.6 * powEta5),
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

    res2 = tf.convert_to_tensor([(0.1 - 0.1 * eta - 1.2 * powEta2 + 1.2 * powEta3 + 1.6 * powEta4 - 1.6 * powEta5),
        3.2 * powEta5 - 2.588854381999832 * powEta4 - 3.505572809000084 * powEta3 +
        2.83606797749979 * powEta2 + 0.3055728090000842 * eta - 0.247213595499958,
        -3.2 * powEta5 + 0.9888543819998319 * powEta4 + 5.294427190999916 * powEta3 -
        1.63606797749979 * powEta2 - 2.094427190999916 * eta + 0.6472135954999582,
        3.2 * powEta5 + 0.9888543819998319 * powEta4 - 5.294427190999916 * powEta3 -
        1.63606797749979 * powEta2 + 2.094427190999916 * eta + 0.6472135954999582,
        -3.2 * powEta5 - 2.588854381999832 * powEta4 + 3.505572809000084 * powEta3 +
        2.83606797749979 * powEta2 - 0.3055728090000842 * eta - 0.247213595499958,
        (0.1 + 0.1 * eta - 1.2 * powEta2 - 1.2 * powEta3 + 1.6 * powEta4 + 1.6 * powEta5), eta*0, eta*0, eta*0, eta*0, eta*0])

    return res1+res2

def basisSplit(xIn, thisBasis) :
    shape = xIn.get_shape().as_list()

    batchSize = tf.shape(xIn)[0]
    sess = tf.Session()
    batchSizeVal = sess.run(batchSize)

    print('batchSize', batchSize, batchSizeVal)
    #shape2 = tf.shape(xIn).value
    #print('shape2', shape2)
    #ans = tf.size(xIn)
    #rint('ans', ans)
    #x = np.copy(np.tanh(xIn)).flatten()
    #x = tf.reshape(xIn,[-1])
    x = xIn

    final = tf.transpose(thisBasis(x))

    change = final.shape[1]
    print('shape', shape,'change', change, 'final', final)
    shape.append(change)
    shape[0]=-1
    print('lshape',shape)
    return tf.reshape(final, tf.TensorShape(shape))    

#application of polynomial with weights
def func2(x,w) :
    return np.dot(basis2(x),w)