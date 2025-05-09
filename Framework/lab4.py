import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from models_abstracted.perceptron_wrapper import PerceptronModel
from models_abstracted.cross_entropy_wrapper import CrossEntropyModel
from models_abstracted.least_squares_wrapper import LeastSquaresModel
from models_abstracted.softmax_wrapper import SoftmaxModel

print("========== LOADING AND CLEANING DATA ==========")

df = pd.read_csv('./Heart.csv')
features = ['RestBP', 'Chol']
X = df[features].to_numpy(dtype=np.float32)
y = pd.factorize(df['AHD'])[0].astype(np.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_neg = 2 * y_train - 1
y_test_neg = 2 * y_test - 1

print("========== BEGINNING MODEL TRAINING AND TESTING ==========")

models = [
    PerceptronModel(learning_rate=0.01, max_iterations=1000),
    CrossEntropyModel(learning_rate=0.01, epochs=1000),
    LeastSquaresModel(learning_rate=0.01, epochs=1000),
    SoftmaxModel(learning_rate=0.01, epochs=1000)
]

for model in models:
    if hasattr(model, 'requires_negative_labels') and model.requires_negative_labels:
        result = model.evaluate(X_train, y_train_neg, X_test, y_test_neg)
    else:
        result = model.evaluate(X_train, y_train, X_test, y_test)

    test_preds = result['Test Predictions'].flatten()
    test_true = y_test.flatten()
    train_preds = result['Train Predictions'].flatten()
    train_true = y_train.flatten()

    train_misclassified = np.sum(train_preds != train_true)
    test_misclassified = np.sum(test_preds != test_true)

    train_accuracy = 100.0 * np.mean(train_preds == train_true)
    test_accuracy = 100.0 * np.mean(test_preds == test_true)

    print(f"\nModel: {model.name}")
    print(f"Train Accuracy: {train_accuracy:.2f}% | Misclassified: {train_misclassified}")
    print(f"Test Accuracy: {test_accuracy:.2f}% | Misclassified: {test_misclassified}")
