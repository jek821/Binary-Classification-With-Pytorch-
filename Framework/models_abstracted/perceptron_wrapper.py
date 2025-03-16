from models_abstracted.base import BaseModel
import models.perceptron as perceptron
import utils.hyperplane as hyperplane
import numpy as np

class PerceptronModel(BaseModel):
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        super().__init__("Perceptron")
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.hyperplane = None
        self.model = None
        
    def fit(self, X, y):
        # Create hyperplane and model
        self.hyperplane = hyperplane.Hyperplane(dim=X.shape[1])
        self.model = perceptron.Perceptron(learning_rate=self.learning_rate, max_iterations=self.max_iterations)
        
        # Train model
        train_accuracy = self.model.train(self.hyperplane, X, y)
        
        # Add iterations to metrics
        self.metrics['Iterations'] = self.model.num_iterations
        
        return train_accuracy
    
    def predict(self, X):
        if self.model is None or self.hyperplane is None:
            raise RuntimeError("Model must be trained before making predictions")
        return np.array([1 if np.dot(x, self.hyperplane.weights) + self.hyperplane.bias >= 0 else 0 for x in X])