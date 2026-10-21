"""Task 2: TF-IDF + Logistic Regression baselines for category and priority."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

BASE = Path(__file__).parent
PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1, 10],
}


def tune(train, label):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    search = GridSearchCV(pipe, PARAM_GRID, cv=5, scoring="f1_macro")
    search.fit(train["text"], train[label])
    print(f"\n[{label}] best params: {search.best_params_}  (CV macro F1 {search.best_score_:.3f})")
    return search.best_estimator_


def evaluate(model, val, label):
    pred = model.predict(val["text"])
    acc = accuracy_score(val[label], pred)
    f1 = f1_score(val[label], pred, average="macro")
    print(f"[{label}] VALIDATION accuracy {acc:.3f} | macro F1 {f1:.3f}")
    classes = model.classes_
    short = [c[:12] for c in classes]                 # short column names so it fits
    cm = pd.DataFrame(confusion_matrix(val[label], pred, labels=classes),
                      index=[f"true {c}" for c in classes], columns=short)
    print(f"[{label}] confusion matrix (rows = true, columns = predicted):")
    print(cm.to_string())
    return acc, f1


def most_confusing(models, val, n=10):
    """Tickets where the model gave the LOWEST probability to the correct answer."""
    rows = []
    for label, model in models.items():
        proba = model.predict_proba(val["text"])
        classes = list(model.classes_)
        for i, (text, true) in enumerate(zip(val["text"], val[label])):
            p_true = proba[i, classes.index(true)]
            pred = classes[int(np.argmax(proba[i]))]
            rows.append((p_true, label, true, pred, text))
    rows.sort(key=lambda r: r[0])
    print(f"\n{'=' * 60}\n{n} MOST CONFUSING VALIDATION TICKETS\n{'=' * 60}")
    for k, (p, label, true, pred, text) in enumerate(rows[:n], 1):
        print(f"{k:2}. [{label}] true={true} | predicted={pred} | p_true={p:.2f}")
        print(f"    {text[:160]}...")


def main():
    train = pd.read_csv(BASE / "train.csv")
    val = pd.read_csv(BASE / "validation.csv")
    print(f"Train {len(train)} rows | Validation {len(val)} rows")

    models, results = {}, {}
    for label in ["category", "priority"]:
        models[label] = tune(train, label)
        results[label] = evaluate(models[label], val, label)

    most_confusing(models, val)

    for label, model in models.items():
        joblib.dump(model, BASE / f"baseline_{label}.joblib")
    print("\nSaved baseline_category.joblib and baseline_priority.joblib")

    print(f"\n{'=' * 60}\nSUMMARY (validation)\n{'=' * 60}")
    for label, (acc, f1) in results.items():
        print(f"{label.capitalize():9}: accuracy {acc:.3f}, macro F1 {f1:.3f}")


if __name__ == "__main__":
    main()