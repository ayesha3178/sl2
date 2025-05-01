import numpy as np

# Sigmoid Activation and Derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset for XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected output for XOR
y = np.array([[0], [1], [1], [0]])

# Seed for reproducibility
np.random.seed(1)

# Initialize weights and biases
input_layer_neurons = X.shape[1]    # 2 inputs
hidden_neurons = 4                  # You can adjust this
output_neurons = 1

# Weights
wh = np.random.uniform(size=(input_layer_neurons, hidden_neurons))
bh = np.zeros((1, hidden_neurons))
wo = np.random.uniform(size=(hidden_neurons, output_neurons))
bo = np.zeros((1, output_neurons))

# Training process
epochs = 10000
lr = 0.1

for epoch in range(epochs):
    # Forward Propagation
    zh = np.dot(X, wh) + bh
    ah = sigmoid(zh)

    zo = np.dot(ah, wo) + bo
    ao = sigmoid(zo)

    # Compute loss (optional)
    loss = np.mean((y - ao) ** 2)

    # Backward Propagation
    error = y - ao
    d_output = error * sigmoid_derivative(ao)

    error_hidden = d_output.dot(wo.T)
    d_hidden = error_hidden * sigmoid_derivative(ah)

    # Update weights and biases
    wo += ah.T.dot(d_output) * lr
    bo += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hidden) * lr
    bh += np.sum(d_hidden, axis=0, keepdims=True) * lr

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final output
print("\nTrained Output:")
print(np.round(ao, 3))
