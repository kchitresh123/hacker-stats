#!/usr/bin/env python

import numpy as np

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(k)
    knn.fit(X_train, y_train)

    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

plt.title(f"k-NN: Varying Number of Neighbors")
plt.plot(neighbors, test_accuracy, label=f"Testing Accuracy")
plt.plot(neighbors, train_accuracy, label=f"Training Accuracy")
plt.legend()
plt.xlabel(f"Number of Neighbors")
plt.ylabel(f"Accuracy")
plt.show()
