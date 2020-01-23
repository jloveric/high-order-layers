'''
This example does not use the polynomial layers, but instead just tries
to match the function using relu.  I've included this for comparison purposes.
'''
import sys, os
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import snovalleyai_piecewise_polynomial_layers.PolynomialLayers as poly
from tensorflow.keras.layers import *


offset = -0.1
factor = 1.5*3.14159
xTest = (np.arange(100)/50-1.0)
yTest = 0.5*np.cos(factor*(xTest-offset))

xTrain = (tf.random.uniform([1000], minval=-1.0, maxval=1, dtype=tf.float32))
yTrain = 0.5*tf.math.cos(factor*(xTrain-offset))

modelSet = [
    {'name' : '5 hidden units', 'units' : 5},
    {'name' : '10 hidden units', 'units' : 10}, 
    {'name' : '20 hidden units', 'units' : 20},
    #{'name' : '500 hidden units', 'units' : 500}
    ]


colorIndex = ['red', 'green', 'blue', 'purple','black']
symbol = ['+','x','o','v','.']

thisModelSet = modelSet

for i in range(0, len(thisModelSet)) :

    model = tf.keras.models.Sequential([
        Dense(1),
        Dense(thisModelSet[i]['units'],activation='relu'),
        Dense(thisModelSet[i]['units'],activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['accuracy'])

    model.fit(xTrain, yTrain, epochs=40, batch_size=1)
    model.evaluate(xTrain, yTrain)

    predictions = model.predict(xTest)
    
    plt.scatter(xTest,predictions.flatten(),c=colorIndex[i],marker=symbol[i], label=thisModelSet[i]['name'])

plt.plot(xTest,yTest,'-',label='actual', color='black')
plt.title('standard relu layer - two hidden layers')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()