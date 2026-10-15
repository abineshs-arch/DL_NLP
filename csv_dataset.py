import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from sklearn.datasets import load_breast_cancer

CSV_PATH = "breast_cancer.csv"


class CsvDataset(Dataset):
    """Reads a CSV; the last column is the label, all other columns are features."""

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)
        self.mean = torch.zeros(self.X.shape[1])   # no scaling until set_scaling() is called
        self.std = torch.ones(self.X.shape[1])

    def set_scaling(self, mean, std):
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.X)                          # number of rows

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mean) / self.std    # scale one row
        return x, self.y[idx]                       # (features, label)


def make_csv(path=CSV_PATH):
    """Save the breast cancer data as a CSV: 30 feature columns + 'target' last."""
    df = load_breast_cancer(as_frame=True).frame
    df.to_csv(path, index=False)
    return df


def get_datasets(path=CSV_PATH, seed=42):
    """80/20 random split; scaling learned from the TRAIN rows only (no leakage)."""
    ds = CsvDataset(path)
    train_ds, val_ds = random_split(ds, [0.8, 0.2],
                                    generator=torch.Generator().manual_seed(seed))
    train_X = ds.X[train_ds.indices]
    ds.set_scaling(train_X.mean(dim=0), train_X.std(dim=0))
    return ds, train_ds, val_ds