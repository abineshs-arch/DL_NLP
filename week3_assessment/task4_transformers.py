"""Week 3 Assessment - Task 4: tokenizer, pipelines, fine-tuning DistilBERT and comparison."""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, pipeline)

BASE = Path(__file__).parent
MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = BASE / "sentiment_model"
LABELS = ["negative", "positive"]


def part_a_tokenizer_and_pipelines():
    print("=" * 60 + "\nPART A: TOKENIZER\n" + "=" * 60)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    sentences = ["The delivery was fast and the product works great.",
                 "Terrible support, I want a refund."]
    batch = tok(sentences, padding=True)
    for i, s in enumerate(sentences):
        print(f"\n{s}")
        print("  tokens        :", tok.convert_ids_to_tokens(batch["input_ids"][i]))
        print("  input_ids     :", batch["input_ids"][i])
        print("  attention_mask:", batch["attention_mask"][i])

    print("\n" + "=" * 60 + "\nPART A: SENTIMENT PIPELINE\n" + "=" * 60)
    sent = pipeline("sentiment-analysis",
                    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    texts = ["I love this phone, the battery lasts all day.",
             "The app crashed three times today.",
             "Customer service answered within minutes, very helpful."]
    for t, r in zip(texts, sent(texts)):
        print(f"  {r['label']:8} {r['score']:.3f} | {t}")

    print("\n" + "=" * 60 + "\nPART A: NER PIPELINE\n" + "=" * 60)
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    for t in ["Priya from Infosys called the Chennai office on Monday.",
              "Amazon opened a new warehouse in Bengaluru.",
              "Sundar Pichai spoke at Google I/O in California."]:
        ents = ", ".join(f"{e['word']} ({e['entity_group']})" for e in ner(t))
        print(f"  {t}\n    -> {ents}")


def part_b_finetune_and_compare():
    train = pd.read_csv(BASE / "sentiment_train.csv")
    test = pd.read_csv(BASE / "sentiment_test.csv")
    label2id = {l: i for i, l in enumerate(LABELS)}
    rows = []

    # ---------- TF-IDF baseline (same settings as Task 3) ----------
    base = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=1000, C=5))
    start = time.time()
    base.fit(train["text"], train["label"])
    base_train = time.time() - start
    start = time.time()
    pred = base.predict(test["text"])
    base_ms = (time.time() - start) / len(test) * 1000
    rows.append(["TF-IDF + LogReg", accuracy_score(test["label"], pred),
                 f1_score(test["label"], pred, average="macro"), base_train, base_ms])

    # ---------- Fine-tune DistilBERT ----------
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def to_ds(df):
        ds = Dataset.from_pandas(pd.DataFrame({"text": df["text"], "label": df["label"].map(label2id)}),
                                 preserve_index=False)
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=64), batched=True)

    train_ds, test_ds = to_ds(train), to_ds(test)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=dict(enumerate(LABELS)), label2id=label2id)

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        p = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(y, p), "f1": f1_score(y, p, average="macro")}

    args = TrainingArguments(output_dir=str(BASE / "hf_output"), learning_rate=3e-5,
                             num_train_epochs=3, per_device_train_batch_size=16,
                             per_device_eval_batch_size=32, eval_strategy="epoch",
                             logging_strategy="epoch", save_strategy="no",
                             report_to="none", seed=42)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)

    print("\n" + "=" * 60 + "\nPART B: FINE-TUNING DistilBERT (CPU)\n" + "=" * 60)
    start = time.time()
    trainer.train()
    bert_train = time.time() - start
    for log in trainer.state.log_history:
        if "eval_accuracy" in log:
            print(f"Epoch {log['epoch']:.0f}: accuracy {log['eval_accuracy']:.3f} | F1 {log['eval_f1']:.3f}")

    start = time.time()
    out = trainer.predict(test_ds)
    bert_ms = (time.time() - start) / len(test) * 1000
    p = [LABELS[i] for i in out.predictions.argmax(axis=-1)]
    rows.append(["DistilBERT (fine-tuned)", accuracy_score(test["label"], p),
                 f1_score(test["label"], p, average="macro"), bert_train, bert_ms])

    trainer.model.save_pretrained(SAVE_DIR)
    tok.save_pretrained(SAVE_DIR)
    print(f"Saved model to {SAVE_DIR.name}/")

    table = pd.DataFrame(rows, columns=["Model", "Accuracy", "Macro F1",
                                        "Training time (s)", "Inference (ms/text)"])
    print("\n" + "=" * 60 + "\nCOMPARISON TABLE (same test set, 400 texts)\n" + "=" * 60)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    part_a_tokenizer_and_pipelines()
    part_b_finetune_and_compare()