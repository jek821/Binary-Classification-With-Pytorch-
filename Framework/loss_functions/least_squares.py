import numpy as np

class LeastSquaresLoss:
    @staticmethod
    def compute(y_true, y_pred):
      return np.mean(np.log(1 + np.exp(-y_true * z)))
