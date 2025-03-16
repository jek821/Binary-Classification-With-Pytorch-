# models/lp_model.py
from models_abstracted.base import BaseModel
import models.linear_programming_classifier as lp
import numpy as np

class LinearProgrammingModel(BaseModel):
    def __init__(self):
        super().__init__("Linear Programming")
        self.model = None
        
    def fit(self, X, y):
        # Create and train model
        self.model = lp.LinearProgramClassifier()
        self.model.fit(X, y)
        
        # Store additional metrics
        self.metrics['Iterations'] = 'N/A'  # LP doesn't have iterations
        
        # Return training accuracy
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)