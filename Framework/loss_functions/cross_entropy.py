import numpy as np

class CrossEntropyLoss:
    @staticmethod
    def compute(y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0) issues
        
        # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))