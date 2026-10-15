import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from csv_dataset import make_csv, get_datasets, CSV_PATH
from mlp import MLP, evaluate


def predict(features, model, ds):
    """One raw row (30 numbers) -> (label, probability of benign)."""
    x = torch.tensor(features, dtype=torch.float32)
    x = (x - ds.mean) / ds.std                      # same scaling as training
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x.unsqueeze(0))).item()
    label = "benign" if prob >= 0.5 else "malignant"   # in this dataset 1 = benign
    return label, round(prob, 3)


def main():
    if not os.path.exists(CSV_PATH):
        make_csv()
    torch.manual_seed(42)
    ds, train_ds, val_ds = get_datasets()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # ---- Train for 30 epochs ----
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(1, 31):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss {loss.item():.3f} | val accuracy {evaluate(model, val_loader):.3f}")

    acc_before = evaluate(model, val_loader)

    # ---- Save only the weights ----
    torch.save(model.state_dict(), "model.pt")

    # ---- Load into a brand-new model object ----
    loaded = MLP()
    loaded.load_state_dict(torch.load("model.pt"))
    loaded.eval()
    acc_after = evaluate(loaded, val_loader)

    print(f"\nValidation accuracy BEFORE saving: {acc_before:.4f}")
    print(f"Validation accuracy AFTER loading: {acc_after:.4f}")
    print(f"Identical: {acc_before == acc_after}")

    # ---- predict() for one new raw row ----
    df = pd.read_csv(CSV_PATH)
    raw_row = df.iloc[0, :-1].values
    actual = "benign" if df.iloc[0, -1] == 1 else "malignant"
    print(f"\npredict(row 0) -> {predict(raw_row, loaded, ds)} | actual: {actual}")


if __name__ == "__main__":
    main()