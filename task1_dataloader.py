import os
import time
import torch
from torch.utils.data import DataLoader
from csv_dataset import CsvDataset, make_csv, get_datasets, CSV_PATH


def main():
    torch.manual_seed(42)
    if not os.path.exists(CSV_PATH):
        make_csv()
        print(f"Created {CSV_PATH}")

    # ---- Full dataset in batches of 32 ----
    ds = CsvDataset(CSV_PATH)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    xb, yb = next(iter(loader))
    n_batches = sum(1 for _ in loader)
    print(f"Dataset size: {len(ds)} | batch features: {xb.shape} | "
          f"batch labels: {yb.shape} | batches: {n_batches}")

    # ---- Train / validation split ----
    ds, train_ds, val_ds = get_datasets()
    print(f"Train {len(train_ds)} rows | Validation {len(val_ds)} rows")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    count = sum(1 for _ in train_loader)
    print(f"One epoch over the train loader = {count} batches")

    # ---- num_workers: 0 vs 2 ----
    for nw in [0, 2]:
        dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=nw)
        start = time.time()
        for _ in range(3):
            for xb, yb in dl:
                pass
        print(f"num_workers={nw}: 3 epochs in {time.time() - start:.2f} s")


if __name__ == "__main__":      # required on Windows when num_workers > 0
    main()