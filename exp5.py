import numpy as np

# Define bipolar training pairs
X = np.array([[1, -1, 1],
              [-1, -1, 1]])

Y = np.array([[1, -1],
              [-1, 1]])

# Step 1: Compute weight matrix W using Hebbian learning
W = np.zeros((X.shape[1], Y.shape[1]))

for i in range(len(X)):
    W += np.outer(X[i], Y[i])

print("Weight Matrix W:")
print(W)

# Bipolar sign activation function
def bipolar_sign(x):
    return np.where(x >= 0, 1, -1)

# Step 2: Recall function from X to Y
def recall_Y(x_input):
    y_output = bipolar_sign(np.dot(x_input, W))
    return y_output

# Step 3: Recall function from Y to X
def recall_X(y_input):
    x_output = bipolar_sign(np.dot(W, y_input))
    return x_output

# Test recall in both directions
test_x = X[0]
recalled_y = recall_Y(test_x)
recalled_x = recall_X(recalled_y)

print("\nTest Input X:", test_x)
print("Recalled Y:", recalled_y)
print("Recalled back X:", recalled_x)
