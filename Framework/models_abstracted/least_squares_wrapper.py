from models_abstracted.base import BaseModel
import utils.hyperplane as hyperplane
from optimization.gradient_descent import GradientDescent
import models.least_squares_model as ls_model
import numpy as np
import time
import tracemalloc

class LeastSquaresModel(BaseModel):
    def __init__(self, learning_rate=0.01, epochs=1000):
        super().__init__("Least Squares")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hyperplane = None
        self.model = None

    def fit(self, X, y):
        y = y.astype(int)
        self.hyperplane = hyperplane.Hyperplane(dim=X.shape[1], activation_function="sigmoid")
        optimizer = GradientDescent(learning_rate=self.learning_rate)
        self.model = ls_model.LeastSquaresModel(self.hyperplane, optimizer)
        self.model.train(X, y, learning_rate=self.learning_rate, epochs=self.epochs)
        self.metrics['Iterations'] = self.epochs
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def predict(self, X):
        if self.model is None or self.hyperplane is None:
            raise RuntimeError("Model must be trained before making predictions")
        y_pred = self.hyperplane.classify(X)
        return (y_pred > 0.5).astype(int)

    def evaluate(self, X_train, y_train, X_test, y_test):
        tracemalloc.start()
        start = time.time()

        self.fit(X_train, y_train)
        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)

        train_acc = 100.0 * np.mean(y_pred_train == y_train)
        test_acc = 100.0 * np.mean(y_pred_test == y_test)

        end = time.time()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            'Model': self.name,
            'Training Accuracy': train_acc,
            'Testing Accuracy': test_acc,
            'Train Predictions': y_pred_train,
            'Test Predictions': y_pred_test,
            'Time (s)': end - start,
            'Memory (MB)': peak / 10**6
        }
