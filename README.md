# Artificial-Neural-Networks
MNIST database training
MNIST Neural Network with PyTorch
Project Overview

This project implements a simple feed-forward neural network to classify handwritten digits from the MNIST dataset using PyTorch.

The goal of this exercise was to understand the core workflow of training neural networks, including:

Dataset loading

Model architecture design

Training with gradient descent

Evaluating model performance

Visualizing learning behavior

The model learns to classify digits 0–9 from grayscale images of size 28 × 28 pixels.

Key Concepts Demonstrated
1. Neural Network Architecture

The model is implemented using a fully connected neural network.

Architecture:

Input Layer: 784 neurons (28×28 pixels flattened)
Hidden Layer 1: 512 neurons + ReLU
Hidden Layer 2: 512 neurons + ReLU
Output Layer: 10 neurons (digit classes 0–9)

The network outputs logits, which are converted to probabilities internally by the loss function.

Technologies Used

Python

PyTorch

Torchvision

Matplotlib

NumPy

Project Workflow
1. Load and Prepare Dataset

The MNIST dataset is loaded using torchvision.datasets.

Each image:

is normalized

converted into tensors

provided through a DataLoader for batch training.

Training set: 60,000 images  
Test set: 10,000 images
2. Build the Neural Network

The neural network is defined using a PyTorch nn.Module class.

Key components:

nn.Linear layers

ReLU activation functions

torch.flatten() to convert image matrices into vectors

Example flow inside the model:

Input image (28×28)
      ↓
Flatten to 784
      ↓
Linear Layer
      ↓
ReLU Activation
      ↓
Linear Layer
      ↓
ReLU Activation
      ↓
Output Layer (10 classes)
3. Define Loss Function and Optimizer

Loss function:

CrossEntropyLoss

This is commonly used for multi-class classification problems.

Optimizer:

Stochastic Gradient Descent (SGD)

Learning rate:

0.01

The optimizer updates model weights using backpropagation.

4. Training Loop

For each batch of training data:

Forward pass

Compute loss

Clear previous gradients

Backpropagation

Update weights

Pseudo workflow:

prediction = model(X)
loss = loss_function(prediction, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
5. Model Evaluation

The test dataset is used to evaluate performance after each epoch.

Metrics tracked:

Test Loss

Classification Accuracy

Evaluation runs with gradients disabled using:

torch.no_grad()

This improves efficiency during testing.

6. Visualization

Training progress is visualized using Matplotlib.

Two plots are generated:

Loss Curve

Shows how the model improves over time.

Training Loss
Validation Loss
Accuracy Curve
Training Accuracy
Validation Accuracy

These plots help diagnose:

underfitting

overfitting

learning stability

Example Output

Typical results after a few epochs:

Train Accuracy: ~95–97%
Test Accuracy: ~96–98%

This demonstrates that even a simple neural network can perform well on MNIST.

Key Lessons Learned

This project helped reinforce several important machine learning concepts:

How neural networks learn via gradient descent

The importance of training vs evaluation modes

Managing GPU vs CPU devices

Understanding the training loop in PyTorch

Tracking and visualizing model performance over time
