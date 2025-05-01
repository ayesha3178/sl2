import numpy as np
import matplotlib.pyplot as plt

# Input values
x = np.linspace(-10, 10, 1000)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# Plotting
plt.figure(figsize=(10, 8))

plt.subplot(2, 3, 1)
plt.plot(x, sigmoid(x), label="Sigmoid", color='blue')
plt.title("Sigmoid")
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x, tanh(x), label="Tanh", color='orange')
plt.title("Tanh")
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x, relu(x), label="ReLU", color='green')
plt.title("ReLU")
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x, leaky_relu(x), label="Leaky ReLU", color='purple')
plt.title("Leaky ReLU")
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(x, elu(x), label="ELU", color='red')
plt.title("ELU")
plt.grid(True)

plt.tight_layout()
plt.suptitle("Common Activation Functions in Neural Networks", fontsize=14, y=1.02)
plt.show()
