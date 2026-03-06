# Example of using L2 regularization in a linear regression model
from sklearn.linear_model import Ridge
import numpy as np

# Generating some synthetic data: X are features and y is the target
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X.squeeze() + np.random.randn(100) * 0.1

# Create and train a Ridge regression model
ridge_model = Ridge(alpha=1.0)  # alpha is the regularization strength
ridge_model.fit(X, y)

print("Coefficients:", ridge_model.coef_)