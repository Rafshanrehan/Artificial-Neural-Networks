import torch
import torch.nn as nn

# Mean Squared Error (MSE) Example
y_true = torch.tensor([3.0, -0.5, 2.0])
y_pred = torch.tensor([2.5, 0.0, 2.1])

mse_loss = nn.MSELoss()
mse = mse_loss(y_pred, y_true)
print("Mean Squared Error (MSE):", round(mse.item(), 2))

# Cross-Entropy Loss Example
# Define the target values (one-hot encoded)
y1_true = torch.tensor([[1, 0],
[0, 1],
[1, 0]], dtype=torch.float32)

# Define the predicted probabilities
y1_pred = torch.tensor([[0.8, 0.2],
[0.3, 0.7],
[0.6, 0.4]], dtype=torch.float32)
# Create the loss function (BCE stands for Binary Cross-Entropy)
loss_fn = nn.BCELoss()

# Calculate the loss
bce_loss = loss_fn(y1_pred, y1_true)
#print("Tensor:", bce_loss, type(bce_loss))
print("Cross-Entropy Loss:", round(bce_loss.item(), 3))