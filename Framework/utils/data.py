import numpy as np
from . import hyperplane as hp
import pandas as pd
from . import normalization_utility as Normalization  # Import the normalization class

class Data:
    def __init__(self):
        # Initializes an empty dataset object
        self.X = None  # Feature matrix
        self.y = None  # Labels
    
    def generate_linearly_separable(self, n_samples=100, dim=2, separation=4.0):

        n_half = n_samples // 2  # Split evenly

        # Create well-separated clusters
        X_class_0 = np.random.randn(n_half, dim) - separation
        X_class_1 = np.random.randn(n_half, dim) + separation

        # Stack into dataset
        X = np.vstack((X_class_0, X_class_1))
        y = np.hstack((np.zeros(n_half), np.ones(n_half)))

        # Shuffle to avoid ordering bias
        indices = np.random.permutation(n_samples)
        self.X, self.y = X[indices], y[indices]

        return self.X, self.y

    def generate_non_linearly_separable(self, n_samples=100, dim=2):
        # Generates non-linearly separable data (e.g., XOR pattern)
        self.X = np.random.randn(n_samples, dim)
        self.y = np.sign(np.sin(5 * self.X[:, 0]) + np.cos(5 * self.X[:, 1]))
        return self.X, self.y
    
        
    def partition_data(self, train_ratio=0.8):
        if self.X is None or self.y is None:
            raise ValueError("No data available. Generate or load data first.")

        n_samples = len(self.X)
        n_train = int(n_samples * train_ratio)

        # Ensure X and y are properly aligned before shuffling
        combined = np.hstack((self.X, self.y.reshape(-1, 1)))  # Stack features and labels together
        np.random.shuffle(combined)  # Shuffle rows together
        
        # Split again after shuffling
        X_shuffled = combined[:, :-1]
        y_shuffled = combined[:, -1]

        X_train, X_test = X_shuffled[:n_train], X_shuffled[n_train:]
        y_train, y_test = y_shuffled[:n_train], y_shuffled[n_train:]

        return X_train, X_test, y_train, y_test
    

