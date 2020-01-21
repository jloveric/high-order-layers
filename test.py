import unittest
from snovalleyai_piecewise_polynomial_layers.PolynomialLayers import *
import numpy as np

class TestStringMethods(unittest.TestCase):

    def test_basis(self):
        #Basic test that there is no crash. Polynomials were already 
        # verified in the c code, but move some
        # of those here next.
        b0([[1.0]])

        b1(np.array([[1.0,2.0]]))
        b1D(np.array([[1.0]]))
        b1C(np.array([[1.0]]))

        b2(np.array([[1.0]]))
        b2C(np.array([[1.0]]))
        b2D(np.array([[1.0]]))

        b3(np.array([[1.0]]))
        b4(np.array([[1.0]]))

        b5(np.array([[1.0]]))
        b5C(np.array([[1.0]]))
        b5D(np.array([[1.0]]))
        
        #self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()