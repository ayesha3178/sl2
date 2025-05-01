import numpy as np

# Step 1: Create ASCII binary inputs for digits 0-9
def to_ascii_binary(char):
    return list(map(int, format(ord(char), '07b')))

# Digits 0 to 9 in ASCII
digits = [str(i) for i in range(10)]
X = np.array([to_ascii_binary(d) for d in digits])  # shape: (10, 7)

# Step 2: Define labels (0 for even, 1 for odd)
y = np.array([int(d) % 2 for d in digits])  # 0 for even, 1 for odd

# Step 3: Perceptron training
def train_perceptron(X, y, lr=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        for xi, target in zip(X, y):
            linear_output = np.dot(xi, weights) + bias
            prediction = 1 if linear_output >= 0 else 0
            error = target - prediction
            weights += lr * error * xi
            bias += lr * error
    return weights, bias

# Train the perceptron
weights, bias = train_perceptron(X, y)

# Step 4: Test the perceptron
def predict(x, weights, bias):
    return 1 if np.dot(x, weights) + bias >= 0 else 0

print("Digit | Prediction (0=Even, 1=Odd)")

for i, xi in enumerate(X):
    result = predict(xi, weights, bias)
    print(f"  {i}   |      {result}")
