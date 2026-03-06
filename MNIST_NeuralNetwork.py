'''Step 1: Import Required Libraries'''
import torch
import torchvision #A package in PyTorch that provides utilities, datasets, and models specifically designed for computer vision tasks.
                   
import torchvision.transforms as transforms
                   #Contains various image transformation utilities that can be used 
                   #to preprocess the data (e.g., resizing, cropping, normalizing, 
                   #flipping, etc.).
                   
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

'''Step 2: Load and Prepare the MNIST Dataset'''

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5, ))])
#The above line of code defines a data transformation pipeline using Pytorch's torchvision.transforms module.
#transforms.Compose() is used to combine multiple transformations into one sequential pipeline. 
#It first converts the image to a tensor, then normalizes it (which is standard data preparation). Mean and std dev is 0.5 for grayscale images. 

#Now we load the MNIST dataset from torchvision
train_set = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transform)
test_set = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transform)

train_loader = DataLoader(train_set, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_set, batch_size = 64, shuffle = False)
#dataloader acts as an iterator over the dataset, providing batches of data to the model during training or testing. 
#train_loader will load the training data in batches of 64 samples, which are randomly shuffled at the start of each epoch to avoid bias by order of data.
#test_loader loads test data in batches of 64 samples without shuffling, so that during evaluation the model is consistent for each batch.

'''Step 3: Define the Neural Network Architecture'''
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__() #this initializes the parent class
        self.flatten = nn.Flatten() #this converts tensor into 1D vector
        
        #Sequential is a PyTorch Container Module that allows you to define a series of layers that will be applied in the listed order. 
        #Here, it's a stack of fully connected layers and activation functions
        self.linear_relu_stack = nn.Sequential(nn.Linear(28*28, 128), #image input size to 128 hidden nodes
                                               nn.ReLU(),  
                                               nn.Linear(128, 64), #128 hidden nodes to 64 hidden nodes
                                               nn.ReLU(),
                                               nn.Linear(64,10)) #64 hidden nodes to 10 output nodes
        
     #define forward pass (how data flows through the network)
    def forward(self, x):
        x = self.flatten(x) #flatten image
        logits = self.linear_relu_stack(x) #pass through layers
        return logits #return raw scores
        
#instantiate the model
model = NeuralNetwork() 
# Automatic device selection: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# This ensures model and data are on the same device for computation
model = model.to(device)

print(model)    

'''Step 4: Define Loss Function and Optimizer'''
loss_fn = nn.CrossEntropyLoss() #suitable for multi class classification
optimizer = optim.Adam(model.parameters(), lr=0.001) #adaptive optimizer

'''Step 5: Train The Model'''
def train(dataloader, model, loss_fn, optimizer): 
    size = len(dataloader.dataset)       # Total number of samples in the training set
    train_loss, correct = 0, 0           # Initialize accumulators for total loss and correct predictions

    # Loop over each batch in the training data
    for batch, (X, y) in enumerate(dataloader):
        # Move input data (X) and labels (y) to the correct device (CPU or GPU)
        # This is important if you're using a GPU for faster computation
        X, y = X.to(device), y.to(device)

        # Forward pass: compute model predictions for the batch
        pred = model(X)

        # Compute loss for this batch
        loss = loss_fn(pred, y)

        # Backpropagation steps
        optimizer.zero_grad()     # Reset gradients from previous step
        loss.backward()           # Compute gradients of loss w.r.t. model parameters
        optimizer.step()          # Update model parameters using the optimizer

        # Accumulate total loss (for plotting or reporting later)
        train_loss += loss.item()  # .item() converts tensor to Python float

        # Count number of correct predictions in this batch
        # pred.argmax(1) gives the index of the highest predicted score (predicted class)
        # Compare with true labels y → gives a boolean tensor
        # .type(torch.float) converts True/False to 1.0/0.0
        # .sum().item() sums correct predictions and converts to Python number
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Average loss over all samples (useful for plotting or tracking over epochs)
    train_loss /= size

    # Compute accuracy in percentage
    accuracy = 100 * correct / size

    # Return average loss and accuracy so you can save or plot them
    return train_loss, accuracy


'''Step 6: Evaluate The Model'''
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)       # Total number of samples in the test set
    num_batches = len(dataloader)        # Total number of batches (needed for averaging loss)
    test_loss, correct = 0, 0            # Initialize accumulators for test loss and correct predictions

    # Disable gradient computation: we don't need gradients for evaluation
    # Makes evaluation faster and uses less memory
    with torch.no_grad():
        # Loop over each batch in the test data
        for X, y in dataloader:
            # Move data to the device (CPU or GPU)
            X, y = X.to(device), y.to(device)

            # Forward pass: compute model predictions for the batch
            pred = model(X)

            # Accumulate total loss (sum over batches)
            test_loss += loss_fn(pred, y).item()

            # Count number of correct predictions
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Average loss over all batches (different from training: sum divided by number of batches)
    test_loss /= num_batches

    # Compute overall accuracy in percentage
    accuracy = 100 * correct / size

    # Return test loss and accuracy so you can save them for plotting or monitoring
    return test_loss, accuracy


'''Step 7: Visualizing the Loss and Accuracy'''
epochs = 5  # Number of times to iterate over the entire training dataset

# Lists to store loss and accuracy for each epoch (useful for plotting later)
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

# Loop over each epoch
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")  # Print epoch number


    # Step 1: Train the model for one epoch
    # Returns average training loss and training accuracy for this epoch
    train_loss, train_accuracy = train(train_loader, model, loss_fn, optimizer)


    # Step 2: Evaluate the model on the test set
    # Returns average test loss and test accuracy for this epoch
    test_loss, test_accuracy = test(test_loader, model, loss_fn)

    # Step 3: Store metrics for plotting
    train_losses.append(train_loss)          # Track training loss over epochs
    test_losses.append(test_loss)            # Track test loss over epochs
    train_accuracies.append(train_accuracy)  # Track training accuracy over epochs
    test_accuracies.append(test_accuracy)    # Track test accuracy over epochs


    # Step 4: Print metrics for this epoch
    # {:.4f} → float with 4 decimal places
    # {:.2f}% → float as percentage with 2 decimal places
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
'''Step 8: Plotting the Results'''
plt.figure(figsize=(12, 5))

# Plotting training and validation loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()