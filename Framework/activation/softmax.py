import numpy as np

class Softmax:
    @staticmethod
    def activate(x):
        """
        Softmax activation function that ensures proper handling of binary classification.
        """
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # For binary classification, explicitly create two outputs
        if x.shape[1] == 1:
            x = np.hstack([np.zeros_like(x), x])  # Class 0 and Class 1 logits

        # Apply softmax
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
