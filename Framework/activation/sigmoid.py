import numpy as np

class Sigmoid:
    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(-x))
    
