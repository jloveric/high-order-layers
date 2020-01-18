'''
This example is meant to demonstrate how you can map complex
functions using a single input and single output with polynomial
synaptic weights
'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PolynomialLayers as poly

factor = 1.5*3.14159
xTest = np.arange(100)/50-1.0
yTest = 0.5*np.sin(factor*xTest)

xTrain = tf.random.uniform([1000], minval=-1, maxval=1, dtype=tf.float32)
yTrain = 0.5*tf.math.sin(factor*xTrain)

modelSetD = [
    {'name' : 'Discontinuous 1', 'func' : poly.b1D},
    {'name' : 'Discontinous 2', 'func' : poly.b2D}, 
    {'name' : 'Discontinuous 5', 'func' : poly.b5D}]

modelSetC = [
    {'name' : 'Continuous 1', 'func' : poly.b1C},
    {'name' : 'Continous 2', 'func' : poly.b2C}, 
    {'name' : 'Continuous 5', 'func' : poly.b5C}]

modelSet = [
    {'name' : 'Continuous 1', 'func' : poly.b1},
    {'name' : 'Continous 2', 'func' : poly.b2},
    {'name' : 'Continous 3', 'func' : poly.b3},
    {'name' : 'Continous 4', 'func' : poly.b4}, 
    {'name' : 'Continuous 5', 'func' : poly.b5}]

colorIndex = ['red', 'green', 'blue', 'purple','black']
symbol = ['+','x','o','v','.']

thisModelSet = modelSet

for i in range(0, len(thisModelSet)) :

    model = tf.keras.models.Sequential([
    poly.Polynomial(1, 1,basis=thisModelSet[i]['func']),
    ])

    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['accuracy'])

    model.fit(xTrain, yTrain, epochs=10, batch_size=1)
    model.evaluate(xTrain, yTrain)

    predictions = model.predict(xTrain)
    
    plt.scatter(xTrain,predictions.flatten(),c=colorIndex[i],marker=symbol[i], label=thisModelSet[i]['name'])

plt.plot(xTest,yTest,'-',label='actual')
plt.title('polynomial synapse - no hidden layers')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()