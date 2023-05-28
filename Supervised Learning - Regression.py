#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Supervised Learning - Regression

def linear_regression(x_train, y_train, learning_rate=0.01, num_iterations=1000):
    num_samples = len(y_train)
    num_features = x_train.shape[1]

    # Initialize weights and bias
    weights = np.zeros(num_features)
    bias = 0

    # Perform gradient descent
    for _ in range(num_iterations):
        # Calculate predictions
        y_pred = np.dot(x_train, weights) + bias

        # Calculate gradients
        dw = (1 / num_samples) * np.dot(x_train.T, (y_pred - y_train))
        db = (1 / num_samples) * np.sum(y_pred - y_train)

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias

# Example usage
data = pd.read_csv('C:/Users/Upend/OneDrive/Desktop/titanic/Medical Price Dataset.csv')

# Prepare the data
x_train = data[['age', 'bmi', 'children']].values
y_train = data['charges'].values

# Normalize the features (optional but recommended)
x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)

# Add bias term to the features
x_train = np.c_[np.ones(len(x_train)), x_train]

# Perform linear regression
weights, bias = linear_regression(x_train, y_train)

# Print the learned parameters
print("Weights:", weights)
print("Bias:", bias)

# Plot the predicted values against the actual values
y_pred = np.dot(x_train, weights) + bias
plt.scatter(y_train, y_pred)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Linear Regression')
plt.show()


# In[ ]:




