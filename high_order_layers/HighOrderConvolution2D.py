from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Layer
import tensorflow as tf
import copy


#Just evaluate the basis function on the input and reshape to the desired form
class ExpansionLayer2D(Layer) :
    def __init__(self, basis=None) :
        super().__init__()
        if basis==None :
            raise Exception('You must define the basis function in ExpansionLayer2D')
        self.basis = basis

    def build(self,input_shape) :
        pass
    
    def call(self, inputs) :
        res = self.basis(inputs)
        res = tf.transpose(res, [1, 2, 3, 4, 0])
        ts = res.get_shape()
        res = tf.reshape(res, [-1,res.shape[1],res.shape[2],res.shape[3]*res.shape[4]])
        return res

def high_order_convolution2D(
            x,
            filters,
            kernel_size,
            basis=None,
            strides=(1, 1),
            padding='valid',
            data_format=None,
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs) :

    x=ExpansionLayer2D(basis=basis)(x)
    x = Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)(x)

    return x

