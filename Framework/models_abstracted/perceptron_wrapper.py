from models_abstracted.base import BaseModel
import models.perceptron as perceptron
import utils.hyperplane as hyperplane
import numpy as np
import time
import tracemalloc

class PerceptronModel(BaseModel):
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        super().__init__("Perceptron")
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.hyperplane = None
        self.model = None
        self.requires_negative_labels = True  # Flag to indicate label format

    def fit(self, X, y):
        self.hyperplane = hyperplane.Hyperplane(dim=X.shape[1])
        self.model = perceptron.Perceptron(learning_rate=self.learning_rate, max_iterations=self.max_iterations)
        train_accuracy = self.model.train(self.hyperplane, X, y)
        self.metrics['Iterations'] = self.model.num_iterations
        return train_accuracy

    def predict(self, X):
        if self.model is None or self.hyperplane is None:
            raise RuntimeError("Model must be trained before making predictions")
        return np.array([1 if np.dot(x, self.hyperplane.weights) + self.hyperplane.bias >= 0 else -1 for x in X])

    def evaluate(self, X_train, y_train, X_test, y_test):
        tracemalloc.start()
        start = time.time()

        self.fit(X_train, y_train)
        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)

        # Convert back to {0,1} for accuracy comparison
        y_true_train = ((y_train + 1) // 2).astype(int)
        y_true_test = ((y_test + 1) // 2).astype(int)
        y_pred_train_binary = ((y_pred_train + 1) // 2).astype(int)
        y_pred_test_binary = ((y_pred_test + 1) // 2).astype(int)

        train_acc = 100.0 * np.mean(y_pred_train_binary == y_true_train)
        test_acc = 100.0 * np.mean(y_pred_test_binary == y_true_test)

        end = time.time()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            'Model': self.name,
            'Training Accuracy': train_acc,
            'Testing Accuracy': test_acc,
            'Train Predictions': y_pred_train_binary,
            'Test Predictions': y_pred_test_binary,
            'Time (s)': end - start,
            'Memory (MB)': peak / 10**6
        }
