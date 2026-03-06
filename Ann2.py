## Activation Functions Implementation & Visualization (now using PyTorch)

# Import necessary libraries
import torch
import torch.nn as nn # package to create and train ANNs
import torch.nn.functional as F # package to access Torch functions
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(0)

# Example of different activation functions using Torch

# set values for x
x = torch.linspace(-10, 10, 100)

# Sigmoid function
sigmoid = torch.sigmoid(x)
plt.plot(x.numpy(), sigmoid.numpy(), label='Sigmoid') 
                    # x.numpy() convers Tensor x 
                    #into numpy array for plotting

# Tanh function
tanh = torch.tanh(x)
plt.plot(x.numpy(), tanh.numpy() / 2 + 0.5, label='Tanh')

# ReLU function
relu = F.relu(x)
plt.plot(x.numpy(), relu.numpy() / 10, label='ReLU')

# Softmax function
softmax = torch.nn.Softmax(dim=0)
plt.plot(x.numpy(), softmax(x), label='Softmax')

plt.legend()
plt.title('Activation Functions')
plt.show()