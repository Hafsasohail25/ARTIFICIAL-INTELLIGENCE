
# ai_lab_general.py

import numpy as np

# 1. Linear Regression using Gradient Descent
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# 2. A Simple Feedforward Neural Network
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.hidden_layer = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        output = self.sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output)
        return output


# 3. A* Search Algorithm
from queue import PriorityQueue

def a_star_search(graph, start, goal, heuristic):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic[start]

    while not open_set.empty():
        _, current = open_set.get()
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor, weight in graph[current].items():
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic[neighbor]
                open_set.put((f_score[neighbor], neighbor))

    return None


# Example usage
if __name__ == "__main__":
    # Linear Regression Example
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = LinearRegressionGD(learning_rate=0.01, epochs=1000)
    model.fit(X, y)
    print("Linear Regression Predictions:", model.predict(np.array([[6], [7]])))

    # Simple Neural Network Example
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1)
    sample_input = np.array([0.5, 0.8])
    print("Neural Network Output:", nn.forward(sample_input))

    # A* Search Algorithm Example
    graph = {
        'A': {'B': 1, 'C': 3},
        'B': {'A': 1, 'D': 1, 'E': 4},
        'C': {'A': 3, 'F': 5},
        'D': {'B': 1},
        'E': {'B': 4, 'F': 2},
        'F': {'C': 5, 'E': 2}
    }
    heuristic = {'A': 7, 'B': 6, 'C': 2, 'D': 1, 'E': 3, 'F': 0}
    print("A* Path:", a_star_search(graph, 'A', 'F', heuristic))
