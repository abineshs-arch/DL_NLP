import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader):
    """Accuracy of the model on all batches of a DataLoader."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += ((model(xb) > 0).float() == yb).sum().item()
    return correct / len(loader.dataset)