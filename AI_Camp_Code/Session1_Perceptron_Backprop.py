# Session 1: Perceptron & Backpropagation
import numpy as np

# Activation (step) function
def step(x):
    return np.where(x >= 0, 1, 0)

# Perceptron training for AND gate example
class Perceptron:
    def __init__(self, lr=0.1, epochs=10):
        self.lr = lr
        self.epochs = epochs
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                update = self.lr * (yi - step(np.dot(self.w, xi) + self.b))
                self.w += update * xi
                self.b += update
    def predict(self, X):
        return step(np.dot(X, self.w) + self.b)

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
# Train simple multilayer perceptron
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', max_iter=1000)
mlp.fit(X, y)
print("XOR MLP Predictions:", mlp.predict(X))