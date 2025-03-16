import numpy as np
from . import hyperplane as hp
import pandas as pd
from . import normalization_utility as Normalization  # Import the normalization class

class Data:
    def __init__(self):
        # Initializes an empty dataset object
        self.X = None  # Feature matrix
        self.y = None  # Labels
    
    def generate_linearly_separable(self, n_samples=100, dim=2, separation=4.0):
        """
        Generates perfectly linearly separable data.

        Parameters:
        - n_samples (int): Total number of points.
        - dim (int): Feature space dimensions.
        - separation (float): Controls the margin between classes.

        Returns:
        - X (numpy.ndarray): Feature matrix.
        - y (numpy.ndarray): Binary labels.
        """
        n_half = n_samples // 2  # Split evenly

        # Create well-separated clusters
        X_class_0 = np.random.randn(n_half, dim) - separation
        X_class_1 = np.random.randn(n_half, dim) + separation

        # Stack into dataset
        X = np.vstack((X_class_0, X_class_1))
        y = np.hstack((np.zeros(n_half), np.ones(n_half)))

        # Shuffle to avoid ordering bias
        indices = np.random.permutation(n_samples)
        self.X, self.y = X[indices], y[indices]

        return self.X, self.y

    def generate_non_linearly_separable(self, n_samples=100, dim=2):
        # Generates non-linearly separable data (e.g., XOR pattern)
        self.X = np.random.randn(n_samples, dim)
        self.y = np.sign(np.sin(5 * self.X[:, 0]) + np.cos(5 * self.X[:, 1]))
        return self.X, self.y
    
    def load_from_csv(self, file_path, normalize=None):
        # Loads data from a CSV file and stores it in the object.
        # Parameters:
        #   file_path (str): Path to the CSV file.
        #   normalize (str, optional): Normalization method, either 'min_max' or 'z_score'. Default is None.
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove the first column (index) and the target column from processing
        if "" in numerical_cols:
            numerical_cols.remove("")
        
        # Remove the target column from normalization
        if "AHD" in numerical_cols:
            numerical_cols.remove("AHD")
        
        # Only normalize numerical columns (excluding target)
        norm = Normalization.Normalization()
        if normalize == 'min_max':
            df[numerical_cols] = norm.min_max_normalize(df[numerical_cols])
        elif normalize == 'z_score':
            df[numerical_cols] = norm.z_score_normalize(df[numerical_cols])
        
        # Split into features and target
        self.X = df.iloc[:, 1:-1]  # All columns except first (index) and last (target)
        self.y = df.iloc[:, -1].values  # Last column as labels
        
        # Convert categorical features to numerical using one-hot encoding
        # This could be replaced with other encoding methods as needed
        if len(categorical_cols) > 0:
            self.X = pd.get_dummies(self.X, columns=[col for col in categorical_cols if col in self.X.columns])
        
        return self.X.values, self.y
        
    def partition_data(self, train_ratio=0.8):
        if self.X is None or self.y is None:
            raise ValueError("No data available. Generate or load data first.")

        n_samples = len(self.X)
        n_train = int(n_samples * train_ratio)

        # Ensure X and y are properly aligned before shuffling
        combined = np.hstack((self.X, self.y.reshape(-1, 1)))  # Stack features and labels together
        np.random.shuffle(combined)  # Shuffle rows together
        
        # Split again after shuffling
        X_shuffled = combined[:, :-1]
        y_shuffled = combined[:, -1]

        X_train, X_test = X_shuffled[:n_train], X_shuffled[n_train:]
        y_train, y_test = y_shuffled[:n_train], y_shuffled[n_train:]

        return X_train, X_test, y_train, y_test
