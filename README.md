# Week 3 Day 3 - Dataset, DataLoader, saving and loading

Run each script with VS Code's Run button (or `py <file>.py`):
- `task1_dataloader.py` - Dataset, DataLoader, random_split, num_workers
- `task2_train_save_load.py` - train 30 epochs, save/load state_dict, predict()
- `csv_dataset.py` (CsvDataset class) and `mlp.py` (MLP model) are imported by the scripts

## Results
- Dataset size: 569 | batch features: torch.Size([32, 30]) | batches: 18
- random_split 80/20: 456 train / 113 validation; one training epoch = 15 batches. Scaling computed from training rows only.
- Validation accuracy after 30 epochs: 0.9469 (107 of 113)
- Before saving 0.9469 = after loading 0.9469 -> Identical: True
- predict(row 0) -> ('malignant', 0.0); actual: malignant

## num_workers
num_workers is the number of background processes that load batches in parallel while the model trains (0 = the main process loads everything). Here num_workers=0 took 0.04 s for 3 epochs, but num_workers=2 took 16.30 s: starting worker processes on Windows is slow, and this data is tiny and already in memory. Workers help when each item is expensive to load, e.g. images read from disk.

## Why save only the state_dict?
The state_dict holds only the learned weights (tensors), not the Python class. It is small, portable and robust: the model code can be changed or moved and the weights still load. Saving the whole model object (`torch.save(model)`) depends on the exact file and class structure at save time and easily breaks when the code changes.