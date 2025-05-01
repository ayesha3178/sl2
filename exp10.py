
import numpy as np


class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """Train the network using Hebbian learning rule"""
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)  # Remove self-connections
        self.weights /= self.size  # Normalize by number of neurons

    def recall(self, pattern, max_iter=100):
        """Asynchronous pattern recall with random update order"""
        pattern = pattern.copy()
        for _ in range(max_iter):
            prev_pattern = pattern.copy()
            # Update neurons in random order
            for i in np.random.permutation(self.size):
                activation = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if activation >= 0 else -1
            if np.allclose(pattern, prev_pattern):
                break  # Converged
        return pattern


# Define 4 orthogonal vectors (size 4)
vectors = [
    np.array([1, 1, 1, 1]),
    np.array([1, -1, 1, -1]),
    np.array([1, 1, -1, -1]),
    np.array([1, -1, -1, 1])
]

# Create and train network
hn = HopfieldNetwork(size=4)
hn.train(vectors)

# Test recall with noisy input
test_pattern = np.array([1, -1, 1, 1])  # Noisy version of first pattern
print("Input pattern:", test_pattern)
result = hn.recall(test_pattern)
print("Recalled pattern:", result)

# Verify weight matrix
print("\nWeight matrix:")
print(hn.weights)