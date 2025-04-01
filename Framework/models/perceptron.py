import numpy as np

class Perceptron:
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.hyperplane = None
        self.num_iterations = 0  # Track number of iterations before convergence
    
    def train(self, hyperplane, X, y):
        self.hyperplane = hyperplane
        
        # Reset hyperplane weights for perceptron learning
        n_features = X.shape[1]
        self.hyperplane.weights = np.zeros(n_features)
        self.hyperplane.bias = 0
        
        # Training loop
        for _ in range(self.max_iterations):
            mistakes = 0
            self.num_iterations += 1
            
            # Process each sample
            for i in range(len(X)):
                # Get raw output for perceptron decision
                linear_output = np.dot(X[i], self.hyperplane.weights) + self.hyperplane.bias
                prediction = 1 if linear_output >= 0 else 0
                
                # Update weights if prediction is wrong
                if prediction != y[i]:
                    # Perceptron update rule
                    update = self.learning_rate * (y[i] - prediction)
                    self.hyperplane.weights += update * X[i]
                    self.hyperplane.bias += update
                    mistakes += 1
            
            # Stop if perfectly classified
            if mistakes == 0:
                break
        
        # Return final training accuracy
        return self.test(X, y)
    
    def test(self, X, y):
        if self.hyperplane is None:
            raise ValueError("Perceptron has not been trained yet")
        
        # Make predictions
        predictions = []
        for i in range(len(X)):
            linear_output = np.dot(X[i], self.hyperplane.weights) + self.hyperplane.bias
            prediction = 1 if linear_output >= 0 else 0
            predictions.append(prediction)
        
        # Calculate accuracy
        predictions = np.array(predictions)
        accuracy = np.mean(predictions == y)
        
        return accuracy