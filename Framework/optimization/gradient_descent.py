import numpy as np
from activation.sigmoid import Sigmoid
from activation.softmax import Softmax
from loss_functions.cross_entropy import CrossEntropyLoss
from loss_functions.softmax import SoftmaxLoss
from loss_functions.least_squares import LeastSquaresLoss

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def compute_gradient(self, X, y_true, y_pred, activation, loss_function):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        
        if activation == "sigmoid":
            gradient_activation = y_pred * (1 - y_pred)
            error = y_pred - y_true
        elif activation == "softmax":
            # For softmax with cross-entropy, the gradient simplifies to this
            error = y_pred - y_true
            gradient_activation = 1 
        else:
            raise ValueError("Unsupported activation function")
            
        # With softmax+cross-entropy, the gradient_activation is already factored into error
        gradient_w = np.dot(X.T, error * gradient_activation) / len(y_true)
        gradient_b = np.mean(error * gradient_activation)
        
        return gradient_w, gradient_b

    def update_hyperplane(self, hyperplane, X, y_true, loss_function="cross_entropy"):
        linear_output = np.dot(X, hyperplane.weights) + hyperplane.bias
        
        if loss_function == "cross_entropy":
            y_pred = Sigmoid.activate(linear_output)
            loss = CrossEntropyLoss.compute(y_true, y_pred)
            activation = "sigmoid"
        elif loss_function == "softmax":
            y_pred = Softmax.activate(linear_output)
            loss = SoftmaxLoss.compute(y_true, y_pred)
            activation = "softmax"
        elif loss_function == "least_squares":
            y_pred = Sigmoid.activate(linear_output)
            loss = LeastSquaresLoss.compute(y_true, y_pred)
            activation = "sigmoid"
        else:
            raise ValueError("Unsupported loss function")

        gradient_w, gradient_b = self.compute_gradient(X, y_true, y_pred, activation, loss_function)

        hyperplane.weights -= self.learning_rate * gradient_w
        hyperplane.bias -= self.learning_rate * gradient_b
        
        return loss