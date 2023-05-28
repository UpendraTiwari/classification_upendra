#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Supervised Learning - Classification

def naive_bayes(x_train, y_train, x_test):
    # Calculate class probabilities
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    class_probs = class_counts / len(y_train)

    # Calculate feature probabilities for each class
    feature_probs = []
    for c in unique_classes:
        c_indices = np.where(y_train == c)[0]
        c_features = x_train[c_indices]
        c_probs = (c_features.sum(axis=0) + 1) / (c_features.sum() + 2)
        feature_probs.append(c_probs)

    # Calculate the log-likelihood of each class for the test data
    log_likelihoods = []
    for c_probs in feature_probs:
        log_likelihood = np.log(c_probs).sum()
        log_likelihoods.append(log_likelihood)

    # Calculate the posterior probability for each class
    posteriors = []
    for i in range(len(log_likelihoods)):
        prior = np.log(class_probs[i])
        posterior = prior + log_likelihoods[i]
        posteriors.append(posterior)

    # Predict the class with the highest posterior probability
    predicted_class = unique_classes[np.argmax(posteriors)]
    return predicted_class


def knn(x_train, y_train, x_test, k=3):
    distances = np.sqrt(np.sum((x_train - x_test)**2, axis=1))
    sorted_indices = np.argsort(distances)
    k_nearest_labels = y_train[sorted_indices][:k]
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
    predicted_label = unique_labels[np.argmax(counts)]
    return predicted_label


# Load the Titanic dataset
data = pd.read_csv('C:/Users/Upend/OneDrive/Desktop/titanic/train.csv')

# Preprocess the dataset
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
data = data.dropna()
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Split the dataset into features and labels
x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].values
y = data['Survived'].values

# Split the dataset into training and test sets
train_size = int(0.8 * len(x))
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Use Naive Bayes algorithm to predict the test set
nb_predictions = [naive_bayes(x_train, y_train, sample) for sample in x_test]

# Use K-Nearest Neighbors algorithm to predict the test set
knn_predictions = [knn(x_train, y_train, sample) for sample in x_test]

# Evaluate the accuracy of the algorithms
nb_accuracy = np.mean(nb_predictions == y_test)
knn_accuracy = np.mean(knn_predictions == y_test)


# In[5]:


print (nb_accuracy, knn_accuracy)


# In[ ]:




