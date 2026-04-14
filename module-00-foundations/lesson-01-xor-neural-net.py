"""
Module 00 — Lesson 1: Neural Network from Scratch
===================================================
We're teaching a neural net to learn the XOR function.

XOR truth table (what we want the net to learn):
  0 XOR 0 = 0
  0 XOR 1 = 1
  1 XOR 0 = 1
  1 XOR 1 = 0

Architecture:
  Input layer:  2 neurons  (the two bits)
  Hidden layer: 4 neurons  (where the magic happens)
  Output layer: 1 neuron   (the prediction: 0 or 1)

No frameworks. Just NumPy + math.
"""

import numpy as np

# ── Reproducible results ──────────────────────────────────────────────────────
np.random.seed(42)

# ── Training data ─────────────────────────────────────────────────────────────
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])                    # shape (4, 2)

y = np.array([
    [0],
    [1],
    [1],
    [0],
])                    # shape (4, 1)

# ── Activation function: Sigmoid ──────────────────────────────────────────────
# Squashes any number into 0..1 — perfect for probabilities
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)   # neat property of sigmoid: ds/dz = s(1-s)

# ── Initialize weights randomly ───────────────────────────────────────────────
# W1: connects input(2) → hidden(4)
# W2: connects hidden(4) → output(1)
# b1, b2: bias terms (one per neuron in the next layer)

W1 = np.random.randn(2, 4)          # (2 inputs, 4 hidden neurons)
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1)         # (4 hidden neurons, 1 output)
b2 = np.zeros((1, 1))

# ── Hyperparameters ───────────────────────────────────────────────────────────
learning_rate = 0.5
epochs        = 10_000

# ── Training loop ─────────────────────────────────────────────────────────────
print("Training...\n")
for epoch in range(epochs):

    # ── 1. FORWARD PASS ───────────────────────────────────────────────────────
    Z1 = X @ W1 + b1          # linear combo into hidden layer
    A1 = sigmoid(Z1)           # activation of hidden layer

    Z2 = A1 @ W2 + b2         # linear combo into output layer
    A2 = sigmoid(Z2)           # final prediction (0..1)

    # ── 2. LOSS (Mean Squared Error) ──────────────────────────────────────────
    loss = np.mean((y - A2) ** 2)

    # ── 3. BACKWARD PASS (backpropagation) ────────────────────────────────────
    # How much did each weight contribute to the error?
    # We work backwards from output → hidden → input.

    dA2 = -2 * (y - A2) / y.shape[0]        # dLoss/dA2
    dZ2 = dA2 * sigmoid_derivative(Z2)       # dLoss/dZ2

    dW2 = A1.T @ dZ2                         # dLoss/dW2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T                         # dLoss/dA1
    dZ1 = dA1 * sigmoid_derivative(Z1)       # dLoss/dZ1

    dW1 = X.T @ dZ1                          # dLoss/dW1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # ── 4. UPDATE WEIGHTS (gradient descent) ──────────────────────────────────
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print progress every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        print(f"  Epoch {epoch+1:>6} | Loss: {loss:.6f}")

# ── Results ───────────────────────────────────────────────────────────────────
print("\n-- Final Predictions ---------------------------------------------")
print(f"{'Input':<12} {'Target':<10} {'Predicted':<12} {'Rounded'}")
print("-" * 48)
for i in range(4):
    inp    = X[i]
    target = int(y[i][0])
    pred   = A2[i][0]
    print(f"{str(inp):<12} {target:<10} {pred:<12.4f} {round(pred)}")

print("\nDone. The network learned XOR from scratch using only math.")

