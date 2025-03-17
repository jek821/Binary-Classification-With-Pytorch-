import numpy as np
from loss_functions.softmax import SoftmaxLoss
from optimization.gradient_descent import GradientDescent

class SoftmaxModel:
    def __init__(self, hyperplane, optimizer):
        self.hyperplane = hyperplane
        self.loss_function = SoftmaxLoss()
        self.optimizer = optimizer

    def train(self, X_train, y_train, learning_rate, epochs):
        self.optimizer.learning_rate = learning_rate

        for epoch in range(epochs):
            y_pred = self.hyperplane.classify(X_train)
            loss = self.loss_function.compute(y_train, y_pred)
            
            y_pred_binary = (y_pred > 0.5).astype(int)
            train_accuracy = np.mean(y_pred_binary == y_train)

            self.optimizer.update_hyperplane(self.hyperplane, X_train, y_train, loss_function="softmax")

            #print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")

            if train_accuracy == 1.0:
                #print(f"Convergence achieved at epoch {epoch + 1}")
                break  

    def test(self, X_test, y_test):
        y_pred = self.hyperplane.classify(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        accuracy = np.mean(y_pred_binary == y_test)

        return accuracy
