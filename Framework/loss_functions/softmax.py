import numpy as np

class SoftmaxLoss:
    @staticmethod
    def compute_loss(y_true, y_pred):
        loss = np.mean(np.log(1 + np.exp(-y_true * y_pred)))  # Element-wise loss
        return loss