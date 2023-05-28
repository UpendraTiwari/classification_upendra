#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Unsupervised Learning


# Read the dataset
dataset = pd.read_csv('C:/Users/Upend/OneDrive/Desktop/titanic/Iris Dataset.csv')

# Remove the "Species" column and store it for future comparison
species = dataset['Species']
dataset = dataset.drop('Species', axis=1)

# Implement the K-Means Clustering
def k_means_clustering(dataset, k, max_iterations=100):
    # Randomly initialize k centroids
    centroids = dataset[np.random.choice(range(len(dataset)), size=k, replace=False)]

    for _ in range(max_iterations):
        # Calculate distances between each data point and centroids
        distances = np.sqrt(np.sum((dataset - centroids[:, np.newaxis])**2, axis=2))

        # Assign each data point to the closest centroid
        labels = np.argmin(distances, axis=0)

        # Update centroids
        new_centroids = np.array([dataset[labels == i].mean(axis=0) for i in range(k)])

        # If centroids don't change, stop iteration
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

# Principal Component Analysis (PCA)
def pca(dataset):
    # Center the data
    centered_data = dataset - np.mean(dataset, axis=0)

    # Calculate covariance matrix
    covariance_matrix = np.cov(centered_data.T)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Project the data onto the eigenvectors
    projected_data = np.dot(centered_data, sorted_eigenvectors)

    return projected_data, sorted_eigenvalues, sorted_eigenvectors

# Perform K-Means Clustering
k = 3  # Number of clusters
labels, centroids = k_means_clustering(dataset.values, k)

# Perform Principal Component Analysis (PCA)
projected_data, eigenvalues, eigenvectors = pca(dataset.values)

# Visualize PCA results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    projected_data[:, 0],
    projected_data[:, 1],
    projected_data[:, 2],
    c='b',
    marker='o'
)

ax.set_xlabel('1st eigenvector')
ax.set_ylabel('2nd eigenvector')
ax.set_zlabel('3rd eigenvector')

plt.show()

# Print the eigenvalues for the corresponding eigenvectors
print("Eigenvalues:")
for i, eigenvalue in enumerate(eigenvalues):
    print(f"Eigenvalue {i+1}: {eigenvalue}")

# Map species labels to numeric values for coloring
species_mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
species_numeric = species.map(species_mapping)

# Visualize clusters
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plotting the cluster output
axes[0].scatter(dataset.values[:, 0], dataset.values[:, 1], c=labels, cmap='viridis')
axes[0].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Cluster Output')

# Plotting the actual species
axes[1].scatter(dataset.values[:, 0], dataset.values[:, 1], c=species_numeric, cmap='viridis')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].set_title('Actual Species')

plt.show()


# In[ ]:




