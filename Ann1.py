
import numpy as np
import matplotlib.pyplot as plt 
import torch 

#set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

#define activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x/exp_x.sum(axis=0)

#generate x values
x = np.linspace(-10,10,400)

#compute y values
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_softmax = softmax(x)

#plotting the activation functions
plt.figure(figsize=(14,10))

plt.subplot(2, 2, 1)
plt.plot(x,y_sigmoid,label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('y = sigmoid(x)')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x,y_tanh,label='Tanh', color = 'orange' )
plt.title('Tanh Activation Function')
plt.xlabel('x')
plt.ylabel('y = tanh(x)')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(x,y_relu,label='relu', color = 'green' )
plt.title('relu Activation Function')
plt.xlabel('x')
plt.ylabel('y = relu(x)')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x,y_softmax,label='softmax', color = 'red' )
plt.title('softmax Activation Function')
plt.xlabel('x')
plt.ylabel('y = softmax(x)')
plt.grid(True)

plt.tight_layout()
plt.show()
