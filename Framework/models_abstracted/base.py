# models/base_model.py
from abc import ABC, abstractmethod
import time
import tracemalloc
import numpy as np

class BaseModel(ABC):
    def __init__(self, name):
        self.name = name
        self.metrics = {}
    
    @abstractmethod
    def fit(self, X, y):
        """Train the model on data X with labels y"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions for data X"""
        pass
    
    def evaluate(self, X_train, y_train, X_test, y_test):
        """Evaluate model and collect standard metrics"""
        # Start tracking time and memory
        tracemalloc.start()
        start_time = time.time()
        
        # Train model
        self.fit(X_train, y_train)
        
        # Get predictions
        train_preds = self.predict(X_train)
        test_preds = self.predict(X_test)
        
        # Calculate accuracy
        train_accuracy = np.mean(train_preds == y_train)
        test_accuracy = np.mean(test_preds == y_test)
        
        # Collect metrics
        end_time = time.time()
        execution_time = end_time - start_time
        current, peak = tracemalloc.get_traced_memory()
        memory_usage = peak / 1024 / 1024  # Convert to MB
        tracemalloc.stop()
        
        # Store metrics
        self.metrics = {
            'Model': self.name,
            'Training Accuracy': train_accuracy * 100,
            'Testing Accuracy': test_accuracy * 100,
            'Time (s)': execution_time,
            'Memory (MB)': memory_usage,
        }
        
        return self.metrics
    
    def get_metrics(self):
        """Return the collected metrics"""
        return self.metrics