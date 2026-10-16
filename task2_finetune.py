import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from transformers import pipeline

from task1_prepare import build_trainer   # reuse everything from Task 1

SAVE_DIR = "emotion_model"


def main():
    trainer, tok, train_ds, val_ds, labels = build_trainer(epochs=3)

    # 1. Train
    print("\nTraining DistilBERT on CPU... (several minutes)")
    start = time.time()
    trainer.train()
    train_minutes = (time.time() - start) / 60

    # 2. Metrics for each epoch
    print("\n" + "=" * 60)
    print("METRICS PER EPOCH")
    print("=" * 60)
    for log in trainer.state.log_history:
        if "eval_accuracy" in log:
            print(f"Epoch {log['epoch']:.0f}: val_loss {log['eval_loss']:.3f} | "
                  f"accuracy {log['eval_accuracy']:.3f} | F1 {log['eval_f1']:.3f}")
    print(f"Training time: {train_minutes:.1f} minutes")

    # 3. Confusion matrix on the validation set
    pred = trainer.predict(trainer.eval_dataset)
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(axis=-1)
    bert_acc = accuracy_score(y_true, y_pred)
    bert_f1 = f1_score(y_true, y_pred, average="macro")

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (rows = true, columns = predicted)")
    print("=" * 60)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=labels, columns=labels)
    print(cm)

    # 4. Save model + tokenizer, then reload in a pipeline
    trainer.model.save_pretrained(SAVE_DIR)
    tok.save_pretrained(SAVE_DIR)
    print(f"\nSaved model and tokenizer to '{SAVE_DIR}/'")

    clf = pipeline("text-classification", model=SAVE_DIR)
    print("\n" + "=" * 60)
    print("RELOADED PIPELINE - 5 NEW SENTENCES")
    print("=" * 60)
    new_sentences = [
        "I finally got the job offer, I am so happy!",
        "My flight got cancelled and nobody is helping me.",
        "I miss my family so much since I moved away.",
        "I am really nervous about my exam tomorrow.",
        "Wow, I did not expect to see you here!",
    ]
    for s, r in zip(new_sentences, clf(new_sentences)):
        print(f"  {r['label']:9} {r['score']:.3f}  | {s}")

    # 5. TF-IDF + Logistic Regression baseline on the SAME split
    base = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    base.fit(list(train_ds["text"]), list(train_ds["label"]))
    base_pred = base.predict(list(val_ds["text"]))
    base_acc = accuracy_score(list(val_ds["label"]), base_pred)
    base_f1 = f1_score(list(val_ds["label"]), base_pred, average="macro")

    print("\n" + "=" * 60)
    print("COMPARISON (same 2000 train / 500 validation split)")
    print("=" * 60)
    print(f"  Fine-tuned DistilBERT : accuracy {bert_acc:.3f} | F1 {bert_f1:.3f}")
    print(f"  TF-IDF + LogReg       : accuracy {base_acc:.3f} | F1 {base_f1:.3f}")


if __name__ == "__main__":
    main()