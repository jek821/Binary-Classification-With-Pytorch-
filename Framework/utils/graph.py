import matplotlib.pyplot as plt
import numpy as np
import os

class Graph:

    def plot_and_save(self, data, hyperplane, save_path):
        # Plots the entire dataset and the decision boundary of the hyperplane.
        # Ensures that the data remains consistent by using the stored dataset.

        # Get the full dataset (not just training data)
        X, y = data.X, data.y  

        # Create figure
        plt.figure(figsize=(10, 8))

        # Separate data points by class
        X_class_0 = X[y == 0]
        X_class_1 = X[y == 1]

        # Plot class 0 points
        plt.scatter(X_class_0[:, 0], X_class_0[:, 1], color='red', marker='o', label="Class 0")

        # Plot class 1 points
        plt.scatter(X_class_1[:, 0], X_class_1[:, 1], color='blue', marker='x', label="Class 1")

        # Get the weight vector and bias
        if hyperplane.activation_function == "softmax":
            w = hyperplane.weights[:, 1] - hyperplane.weights[:, 0]
            b = hyperplane.bias[1] - hyperplane.bias[0]
        else:
            w = hyperplane.weights
            b = hyperplane.bias

        # Get min and max x values
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

        # Avoid division by zero for vertical line cases
        w1, w2 = w[0], w[1]
        if abs(w2) < 1e-10:
            plt.axvline(x=-b/w1, color='k', linestyle='-')
        else:
            y_min = (-w1 * x_min - b) / w2
            y_max = (-w1 * x_max - b) / w2
            plt.plot([x_min, x_max], [y_min, y_max], 'k-', label="Decision Boundary")

        # Set axis limits
        plt.xlim(x_min, x_max)
        plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

        # Add labels and title
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Decision Boundary with {hyperplane.activation_function.capitalize()} Activation')
        
        # Ensure the legend displays both classes
        plt.legend(loc='upper right')

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save figure
        plt.savefig(save_path)
        #print(f"Graph saved to {save_path}")

    def plot_raw_data(self, data, save_path):
        """
        Plots the raw data points without a decision boundary.
        Used to verify that the ML process is not altering the dataset.
        """
        # Get the full dataset
        X, y = data.X, data.y  # Directly access stored data
        
        # Check if data is available
        if X is None or y is None:
            raise ValueError("No data available. Generate or load data first.")
        
        # Create figure
        plt.figure(figsize=(10, 8))

        # Plot data points
        for i in range(len(X)):
            if y[i] == 0:
                plt.scatter(X[i, 0], X[i, 1], color='red', marker='o', label="Class 0" if i == 0 else "")
            else:
                plt.scatter(X[i, 0], X[i, 1], color='blue', marker='x', label="Class 1" if i == 0 else "")

        # Set axis limits
        plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

        # Add labels and title
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Raw Data Visualization (No ML Influence)')
        plt.legend()

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the figure
        plt.savefig(save_path)
        #print(f"Raw data graph saved to {save_path}")
        