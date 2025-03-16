from models_abstracted.base import BaseModel
import utils.hyperplane as hyperplane
from optimization.gradient_descent import GradientDescent
import models.least_squares_model as ls_model
import numpy as np

class LeastSquaresModel(BaseModel):
    def __init__(self, learning_rate=0.01, epochs=1000):
        super().__init__("Least Squares")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hyperplane = None
        self.model = None
        
    def fit(self, X, y):
        # Ensure y labels are in the right format
        y = y.astype(int)
        
        # Initialize model components
        self.hyperplane = hyperplane.Hyperplane(dim=X.shape[1], activation_function="sigmoid")
        optimizer = GradientDescent(learning_rate=self.learning_rate)
        self.model = ls_model.LeastSquaresModel(self.hyperplane, optimizer)
        
        # Train model
        self.model.train(X, y, learning_rate=self.learning_rate, epochs=self.epochs)
        
        # Store additional metrics
        self.metrics['Iterations'] = self.epochs
        
        # Calculate training accuracy
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def predict(self, X):
        if self.model is None or self.hyperplane is None:
            raise RuntimeError("Model must be trained before making predictions")
        y_pred = self.hyperplane.classify(X)
        return (y_pred > 0.5).astype(int)