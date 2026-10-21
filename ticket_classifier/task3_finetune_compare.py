"""Part 2 - Task 1: fine-tune DistilBERT for category and compare with the TF-IDF baseline."""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

BASE = Path(__file__).parent
MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = BASE / "distilbert_category"
MAX_LEN = 128          # tickets are long; 128 tokens keeps CPU training time reasonable


def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro")}


def main():
    train = pd.read_csv(BASE / "train.csv")
    val = pd.read_csv(BASE / "validation.csv")
    test = pd.read_csv(BASE / "test.csv")
    labels = sorted(train["category"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    print(f"Train {len(train)} | Validation {len(val)} | Test {len(test)} | {len(labels)} categories")

    rows = []          # rows of the comparison table
    per_class = {}     # per-category F1 on the test set

    # ---------------- 1. TF-IDF + Logistic Regression baseline ----------------
    baseline = joblib.load(BASE / "baseline_category.joblib")
    start = time.time()
    clone(baseline).fit(train["text"], train["category"])          # time a fresh training run
    base_train_s = time.time() - start
    for split, df in [("validation", val), ("test", test)]:
        start = time.time()
        pred = baseline.predict(df["text"])
        ms = (time.time() - start) / len(df) * 1000
        rows.append(["TF-IDF + LogReg", split, accuracy_score(df["category"], pred),
                     f1_score(df["category"], pred, average="macro"), base_train_s, ms])
        if split == "test":
            per_class["TF-IDF"] = f1_score(df["category"], pred, labels=labels, average=None)

    # ---------------- 2. Fine-tuned DistilBERT ----------------
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def to_dataset(df):
        ds = Dataset.from_pandas(
            pd.DataFrame({"text": df["text"], "label": df["category"].map(label2id)}),
            preserve_index=False)
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True)

    train_ds, val_ds, test_ds = to_dataset(train), to_dataset(val), to_dataset(test)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels),
        id2label={i: l for l, i in label2id.items()}, label2id=label2id)

    args = TrainingArguments(
        output_dir=str(BASE / "hf_output"),
        learning_rate=5e-5,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)

    print("\nFine-tuning DistilBERT on CPU... (this takes a while)")
    start = time.time()
    trainer.train()
    bert_train_s = time.time() - start

    print("\n" + "=" * 60 + "\nDISTILBERT METRICS PER EPOCH (validation)\n" + "=" * 60)
    for log in trainer.state.log_history:
        if "eval_accuracy" in log:
            print(f"Epoch {log['epoch']:.0f}: val_loss {log['eval_loss']:.3f} | "
                  f"accuracy {log['eval_accuracy']:.3f} | F1 {log['eval_f1']:.3f}")

    for split, df, ds in [("validation", val, val_ds), ("test", test, test_ds)]:
        start = time.time()
        out = trainer.predict(ds)
        ms = (time.time() - start) / len(df) * 1000
        pred = [labels[i] for i in out.predictions.argmax(axis=-1)]
        rows.append(["DistilBERT (fine-tuned)", split, accuracy_score(df["category"], pred),
                     f1_score(df["category"], pred, average="macro"), bert_train_s, ms])
        if split == "test":
            per_class["DistilBERT"] = f1_score(df["category"], pred, labels=labels, average=None)

    trainer.model.save_pretrained(SAVE_DIR)
    tok.save_pretrained(SAVE_DIR)
    print(f"\nSaved fine-tuned model to {SAVE_DIR.name}/")

    # ---------------- 3. Comparison ----------------
    table = pd.DataFrame(rows, columns=["Model", "Split", "Accuracy", "Macro F1",
                                        "Training time (s)", "Inference (ms/ticket)"])
    print("\n" + "=" * 60 + "\nCOMPARISON TABLE (category)\n" + "=" * 60)
    print(table.round(3).to_string(index=False))

    pc = pd.DataFrame(per_class, index=labels)
    pc["Winner"] = np.where(pc["DistilBERT"] > pc["TF-IDF"] + 0.02, "DistilBERT",
                            np.where(pc["TF-IDF"] > pc["DistilBERT"] + 0.02, "TF-IDF", "tie"))
    print("\n" + "=" * 60 + "\nPER-CATEGORY F1 ON TEST (where each model wins)\n" + "=" * 60)
    print(pc.round(3).to_string())


if __name__ == "__main__":
    main()