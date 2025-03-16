import numpy as np

class Sigmoid:
    @staticmethod
    def activate(x):
        # Applies the sigmoid function element-wise.
        return 1 / (1 + np.exp(-x))
    
