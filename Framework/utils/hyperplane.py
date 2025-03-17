import numpy as np
from activation.sigmoid import Sigmoid
from activation.softmax import Softmax

class Hyperplane:
    def __init__(self, dim, activation_function="sigmoid"):
        self.activation_function = activation_function
        # Use the same weight shape for both sigmoid and softmax
        # For binary classification, we only need one set of weights
        self.weights = np.random.randn(dim) * 0.1
        self.bias = 0
        
    def classify(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        
        if self.activation_function == "sigmoid":
            return Sigmoid.activate(linear_output)
        elif self.activation_function == "softmax":
            # For binary classification with softmax, we can represent it as
            # probabilities for two complementary events (class 0 and class 1)
            # Convert to shape expected by softmax and then extract probability for class 1
            logits = np.column_stack([-linear_output, linear_output])
            return Softmax.activate(logits)[:, 1]  # Return P(class=1)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")