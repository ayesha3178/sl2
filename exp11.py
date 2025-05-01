import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Binary classification: Setosa vs others

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Build logistic regression model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(4,))
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=50, verbose=0)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Logistic Regression Accuracy:", accuracy)
# Build a simple Neural Network
nn_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
nn_model.fit(X_train, y_train, epochs=50, verbose=0)

# Evaluate
nn_loss, nn_accuracy = nn_model.evaluate(X_test, y_test)
print("Neural Network Accuracy:", nn_accuracy)
