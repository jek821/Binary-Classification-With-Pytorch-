import numpy as np

class LeastSquaresLoss:
    @staticmethod
    def compute(y_true, y_pred):
      return 0.5 * np.mean((y_true - y_pred) ** 2)
