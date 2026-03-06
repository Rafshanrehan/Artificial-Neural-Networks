import numpy as np 

#input vector
x = np.array([0.5,-0.2])

#weight matrix for hidden layer
W1 = np.array([[0.2, 0.8],
               [-0.5, 0.1],
               [0.7,-0.3]])

#bias vector for hidden layer 
b1 = np.array([0.1,-0.2,0.05])

#calculate z1
z1 = np.dot(W1,x)+b1
print("z1=", z1)

#ReLU activation
a1 = np.maximum(0, z1)
print("a1 relu = ", a1)

#hidden layer to output layer

W2 = np.array([[0.4, -0.6, 0.1],
               [-0.3, 0.8, -0.2]])

# Bias vector for output layer
b2 = np.array([-0.1, 0.2])

# Calculate z2
z2 = np.dot(W2, a1) + b2
print("z2 = ", z2)

# Softmax activation function
def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

y = softmax(z2)
print("y (Softmax) = ", y)