import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3" 

import unittest
from snovalleyai_piecewise_polynomial_layers.PolynomialLayers import *
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


if __name__ == '__main__':
    unittest.main()