"""Week 3 Assessment - Task 2: PyTorch training pipeline on the breast cancer data."""
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
torch.manual_seed(42)


class CsvDataset(Dataset):
    """Wraps a feature matrix and labels so a DataLoader can serve batches."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),                  # raw score (logit); sigmoid is inside the loss
        )

    def forward(self, x):
        return self.net(x)


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = (torch.sigmoid(model(xb)) > 0.5).float()
            correct += (pred == yb).sum().item()
            total += len(yb)
    return correct / total


def main():
    # ---------- data: split, then scale (fit the scaler on TRAIN only) ----------
    data = load_breast_cancer()
    X_train, X_val, y_train, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
    scaler = StandardScaler().fit(X_train)
    X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)

    train_loader = DataLoader(CsvDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(CsvDataset(X_val, y_val), batch_size=32, shuffle=False)
    print(f"Train {len(X_train)} | Validation {len(X_val)} | Features {X_train.shape[1]}")
    xb, yb = next(iter(train_loader))
    print(f"One batch: X {list(xb.shape)}, y {list(yb.shape)}")

    # ---------- model ----------
    model = MLP(X_train.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---------- training loop with early stopping ----------
    epochs, patience = 40, 8
    best_acc, best_epoch, best_state, wait = 0.0, 0, None, 0
    train_losses, val_accs = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()            # 1. reset gradients
            out = model(xb)                  # 2. forward
            loss = loss_fn(out, yb)          # 3. loss
            loss.backward()                  # 4. backward
            optimizer.step()                 # 5. update weights
            running += loss.item() * len(xb)
        train_losses.append(running / len(train_loader.dataset))
        val_accs.append(accuracy(model, val_loader))

        if val_accs[-1] > best_acc:
            best_acc, best_epoch, wait = val_accs[-1], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2}: train loss {train_losses[-1]:.4f} | val acc {val_accs[-1]:.4f}")
        if wait >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"Best val accuracy {best_acc:.4f} at epoch {best_epoch}")

    # ---------- save best weights, reload into a NEW model, compare ----------
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), BASE / "mlp_best.pt")
    reloaded = MLP(X_train.shape[1])
    reloaded.load_state_dict(torch.load(BASE / "mlp_best.pt"))
    reloaded.eval()
    acc_original, acc_reloaded = accuracy(model, val_loader), accuracy(reloaded, val_loader)
    print(f"Original model accuracy: {acc_original:.4f}")
    print(f"Reloaded model accuracy: {acc_reloaded:.4f} "
          f"({'identical' if acc_original == acc_reloaded else 'DIFFERENT'})")

    # ---------- scikit-learn baseline on the same split ----------
    logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    print(f"scikit-learn LogisticRegression accuracy: {accuracy_score(y_val, logreg.predict(X_val)):.4f}")

    # ---------- plot ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(range(1, len(train_losses) + 1), train_losses)
    ax1.set(title="Training loss", xlabel="Epoch", ylabel="BCE loss")
    ax2.plot(range(1, len(val_accs) + 1), val_accs, color="green")
    ax2.axvline(best_epoch, linestyle="--", color="gray", label=f"best epoch {best_epoch}")
    ax2.set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(BASE / "mlp_training.png", dpi=150)
    print("Saved mlp_training.png and mlp_best.pt")


if __name__ == "__main__":
    main()