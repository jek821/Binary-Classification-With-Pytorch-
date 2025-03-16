import numpy as np

class CrossEntropyLoss:
    @staticmethod
    def compute(y_true, y_pred):
        """
        Computes cross-entropy loss for both Sigmoid and Softmax outputs.

        For Sigmoid (binary classification):
        y_true: Binary labels (shape (batch_size,))
        y_pred: Sigmoid output (shape (batch_size,))

        For Softmax (multi-class classification):
        y_true: Can be either one-hot encoded (shape (batch_size, n_classes)) or class indices (shape (batch_size,))
        y_pred: Softmax output (shape (batch_size, n_classes))
        """
        # Ensure numerical stability by adding a small constant inside log
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0) issues

        # Check if we're dealing with sigmoid output (1D) or softmax output (2D)
        if y_pred.ndim == 1:
            # Binary classification with sigmoid
            # Use binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Multi-class classification with softmax
            # If labels are class indices (e.g., [0, 1, 1, 0]), convert to one-hot
            if y_true.ndim == 1:
                return -np.mean(np.log(y_pred[np.arange(len(y_true)), y_true]))
            else:
                return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))