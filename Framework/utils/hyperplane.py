import numpy as np
from activation.sigmoid import Sigmoid
from activation.tanh import Tanh

class Hyperplane:
    def __init__(self, dim, activation_function="sigmoid"):
        self.activation_function = activation_function
        
        if activation_function == "softmax":
            self.weights = np.random.randn(dim, 2) * 0.1
            self.bias = np.zeros(2)
        else:
            self.weights = np.random.randn(dim) * 0.1
            self.bias = 0

    def classify(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        
        if self.activation_function == "sigmoid":
            return Sigmoid.activate(linear_output)
        elif self.activation_function == "tanh":
            return Tanh.activate(linear_output)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")
