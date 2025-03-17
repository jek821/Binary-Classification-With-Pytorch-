import numpy as np

class SoftmaxLoss:
    @staticmethod
    def compute(y_true, y_pred):
        """
        Compute cross-entropy loss for softmax outputs.
        """
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0) issues
        
        # Ensure `y_true` is one-hot encoded
        num_classes = y_pred.shape[1]
        y_true_one_hot = np.eye(num_classes)[y_true.astype(int)]

        # Compute categorical cross-entropy loss
        return -np.mean(np.sum(y_true_one_hot * np.log(y_pred), axis=1))
