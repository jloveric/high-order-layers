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

model = tf.keras.models.Sequential([
  poly.Polynomial(1, 1,basis=poly.b5),
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(xTrain, yTrain, epochs=20, batch_size=1)
model.evaluate(xTrain, yTrain)

predictions = model.predict(xTrain)
print("predictions", predictions.shape, predictions)

plt.plot(xTest,yTest,'-')
plt.scatter(xTrain,predictions.flatten(),c='red',marker='+')
plt.title('5th order polynomial synapse - no hidden layers')
plt.xlabel('x')
plt.ylabel('y')
plt.show()