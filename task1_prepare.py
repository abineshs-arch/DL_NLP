import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from collections import Counter
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

MODEL_NAME = "distilbert-base-uncased"


def load_data():
    """Emotion dataset: take 2500 random rows, split 2000 train / 500 validation."""
    raw = load_dataset("dair-ai/emotion")
    labels = raw["train"].features["label"].names      # sadness, joy, love, anger, fear, surprise
    small = raw["train"].shuffle(seed=42).select(range(2500))
    split = small.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    return split["train"], split["test"], labels


def tokenize_data(train_ds, val_ds):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=64)

    train_tok = train_ds.map(tokenize, batched=True)
    val_tok = val_ds.map(tokenize, batched=True)
    return tok, train_tok, val_tok


def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro")}


def build_trainer(epochs=3):
    train_ds, val_ds, labels = load_data()
    tok, train_tok, val_tok = tokenize_data(train_ds, val_ds)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
    )

    args = TrainingArguments(
        output_dir="hf_output",
        learning_rate=2e-5,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",          # evaluate after every epoch
        logging_strategy="epoch",
        save_strategy="no",             # don't fill the folder with checkpoints
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tok),   # pads each batch to the same length
        compute_metrics=compute_metrics,
    )
    return trainer, tok, train_ds, val_ds, labels


def main():
    trainer, tok, train_ds, val_ds, labels = build_trainer()

    print("\n" + "=" * 60)
    print("DATASET")
    print("=" * 60)
    print(f"Train size: {len(train_ds)}   Validation size: {len(val_ds)}")
    print(f"Labels ({len(labels)}): {labels}")

    for name, ds in [("Train", train_ds), ("Validation", val_ds)]:
        counts = Counter(ds["label"])
        dist = ", ".join(f"{labels[i]} {counts[i]}" for i in range(len(labels)))
        print(f"{name} label distribution: {dist}")

    ex = trainer.train_dataset[0]
    print(f"\nExample text : {ex['text']}")
    print(f"Label        : {labels[ex['label']]}")
    print(f"input_ids    : {ex['input_ids']}")

    print("\n" + "=" * 60)
    print("TRAINER READY")
    print("=" * 60)
    a = trainer.args
    print(f"Model: {MODEL_NAME} with num_labels={trainer.model.config.num_labels}")
    print(f"learning_rate={a.learning_rate}, epochs={a.num_train_epochs}, "
          f"batch={a.per_device_train_batch_size}, eval each epoch")
    print("Metrics: accuracy + F1 (macro)")


if __name__ == "__main__":
    main()