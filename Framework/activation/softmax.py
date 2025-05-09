import numpy as np

class Softmax:
    @staticmethod
    def activate(x):
        # Ensure input is 2D: (n_samples, 1) or (n_samples, n_classes)
        x = np.atleast_2d(x)
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability fix
        softmax = exps / np.sum(exps, axis=1, keepdims=True)
        return softmax.squeeze()  # Flatten back to 1D if needed
