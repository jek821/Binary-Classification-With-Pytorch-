import numpy as np
from scipy.optimize import linprog

class LinearProgramClassifier:
    def __init__(self):
        self.weights = None

    def fit(self, features, labels):
        # Transform labels from {0,1} to {-1,+1}
        transformed_labels = 2 * labels - 1
        num_samples, num_features = features.shape

        # Objective function: We just need to satisfy constraints, so minimize 0
        objective_coeffs = np.zeros(num_features)  # Only optimizing weights

        # Constraint matrix and bounds
        constraint_matrix = np.zeros((num_samples, num_features))
        constraint_bounds = np.ones(num_samples)  # y_i (x_i^T w) >= 1

        for i in range(num_samples):
            constraint_matrix[i] = -transformed_labels[i] * features[i]

        # Solve linear program
        result = linprog(
            c=objective_coeffs,  # Minimize 0
            A_ub=constraint_matrix,
            b_ub=-constraint_bounds,  
            method='highs'
        )

        if not result.success:
            raise RuntimeError(f"Linear program optimization failed: {result.message}")

        self.weights = result.x  # Extract weight vector
        return self

    def predict(self, features):
        decision_values = np.dot(features, self.weights)
        return (decision_values >= 0).astype(int)  # Predict class 0 or 1
