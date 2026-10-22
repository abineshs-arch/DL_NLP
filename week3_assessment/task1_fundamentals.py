"""Week 3 Assessment - Task 1: Neural network fundamentals (NumPy + autograd check)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # save plots to files, no pop-up windows
import matplotlib.pyplot as plt

BASE = Path(__file__).parent

# ---------------- Data ----------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)


# ---------------- Basic functions ----------------
def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())               # subtract max for numerical stability
    return e / e.sum()


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def cross_entropy(y_true, y_pred, eps=1e-12):
    """Binary cross-entropy."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ---------------- 2-4-1 network trained with gradient descent ----------------
def train_xor(lr, epochs=5000, seed=1):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, (2, 4))
    b1 = np.full((1, 4), 0.1)             # small positive bias keeps ReLUs alive
    W2 = rng.normal(0, 1, (4, 1))
    b2 = np.zeros((1, 1))
    n = len(X)
    losses = []
    for _ in range(epochs):
        # forward
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        loss = cross_entropy(y, a2)
        losses.append(loss)
        if not np.isfinite(loss):
            break
        # backward (sigmoid + BCE -> gradient is a2 - y)
        dz2 = (a2 - y) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)              # ReLU derivative
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        # gradient descent step
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    preds = sigmoid(relu(X @ W1 + b1) @ W2 + b2).ravel()
    return losses, preds


def autograd_check():
    import torch
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    x, t = 2.0, 10.0
    loss = (w * x + b - t) ** 2
    loss.backward()
    return w.grad.item(), b.grad.item()


def main():
    print("=" * 60 + "\nPART A: basic functions\n" + "=" * 60)
    print("relu([-2, 0, 3])          =", relu(np.array([-2, 0, 3])))
    print("sigmoid([0, 2])           =", sigmoid(np.array([0, 2.0])).round(3))
    print("softmax([2.0, 1.0, 0.1])  =", softmax([2.0, 1.0, 0.1]).round(3))
    print("mse([1,0],[0.9,0.2])      =", round(mse(np.array([1, 0]), np.array([0.9, 0.2])), 3))
    print("cross_entropy([1,0],[0.9,0.2]) =",
          round(cross_entropy(np.array([1, 0]), np.array([0.9, 0.2])), 3))

    print("\n" + "=" * 60 + "\nPART B: gradient of (w*x + b - t)^2\n" + "=" * 60)
    w, x, b, t = 1, 2, 0, 10
    err = w * x + b - t                   # 1*2 + 0 - 10 = -8
    print(f"By hand: error = {err}, dL/dw = 2*error*x = {2 * err * x}, dL/db = 2*error = {2 * err}")
    wg, bg = autograd_check()
    print(f"Autograd: w.grad = {wg}, b.grad = {bg}")

    print("\n" + "=" * 60 + "\nPART C: 2-4-1 network on XOR (lr=0.5, 5000 epochs)\n" + "=" * 60)
    curves = {}
    for lr in [0.5, 0.01, 5]:
        losses, preds = train_xor(lr)
        curves[lr] = losses
        print(f"lr={lr:<4} loss@100 {losses[100]:.3f} | loss@1000 {losses[1000]:.3f} | "
              f"final {losses[-1]:.4f} | predictions {preds.round(2)}")

    plt.figure(figsize=(9, 5))
    for lr, losses in curves.items():
        label = {0.5: "lr=0.5 (good)", 0.01: "lr=0.01 (too slow)", 5: "lr=5 (unstable)"}[lr]
        plt.plot(losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("XOR 2-4-1 network: effect of learning rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(BASE / "xor_loss_curves.png", dpi=150, bbox_inches="tight")
    print("\nSaved xor_loss_curves.png")


if __name__ == "__main__":
    main()