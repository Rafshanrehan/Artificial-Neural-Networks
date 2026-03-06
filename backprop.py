import torch 
import torch.nn as nn 
import torch.optim as optim #package implementing various optimization algorithms

# Example data: features and labels
# Feature tensor (4 samples, 2 features each)
X = torch.tensor([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8], [1.0, 1.1]], dtype=torch.float32)
# Labels (binary classification)
Y = torch.tensor([[0], [0], [1], [1]], dtype=torch.float32)
print(X)
print(Y)

#Simple Neural netwoek with one hidden layer 
class BinaryClassifier(nn.Module):
    '''defines a new class that inherits from nn.Module, which is the base for all NN in pytorch'''
    def __init__(self):
        super(BinaryClassifier, self).__init__() #this calls the super (parent) class's intiialization
        self.hidden = nn.Linear(2, 10) #now we add our own initialization steps. 2 Inputs goes into 10 nodes
        self.output = nn.Linear(10,1) #10 nodes goes into one output 
        self.sigmoid = nn.Sigmoid() #activation function
        
    def forward(self, x): #this applies the formula of caluclating sigmoid(z = sum(w*x)+b)
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
        
model = BinaryClassifier()

# Loss function
criterion = nn.BCELoss()
# Optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Number of epochs (full passes through the dataset)
epochs = 1000
for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing x to the model
    Y_pred = model(X)

    # Compute loss
    loss = criterion(Y_pred, Y)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() #Resets the gradients of all optimized
                          #
    loss.backward()       #backward() method is called on the output tensor, which 
                          #initiates the gradient computation process.
                          #During the backward pass: PyTorch’s autograd system 
                          #recursively computes the gradients of the loss with 
                          #respect to each tensor in the computation graph, using 
                          #the chain rule. The gradients are stored in the .grad 
                          #attribute of each tensor. The gradients are propagated 
                          #from the output tensor to the input tensors, allowing the 
                          #model’s parameters to be updated.
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# After training, check the model's predictions
with torch.no_grad():  #disable gradient computation, which reduces memory consumption 
                       #and speeds up inference.
    Y_pred = model(X)
    predicted = Y_pred.round()  # Round predictions to 0 or 1
    accuracy = (predicted == Y).float().mean()
    print("Accuracy", accuracy.item())