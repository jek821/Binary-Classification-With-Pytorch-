import numpy as np

class SoftmaxLoss:
    @staticmethod
    def compute(y_true, y_pred):
        # For y_true in {-1, 1} and y_pred as raw scores before activation
        eps = 1e-12
        
        # Implementing L_S(w) = (1/P) * sum(log(1 + exp(-y_i * w^T x_i)))
        # Where y_pred represents w^T x_i
        loss = np.log(1 + np.exp(-y_true * y_pred))
        return np.mean(loss)