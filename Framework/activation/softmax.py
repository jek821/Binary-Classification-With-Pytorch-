import numpy as np

class Softmax:
    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(-x))
