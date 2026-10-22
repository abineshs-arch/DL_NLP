"""Week 3 Assessment - Task 3: classical NLP baseline (TF-IDF) and GloVe embeddings."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

BASE = Path(__file__).parent
GLOVE_CACHE = BASE.parent / "glove50.npz"     # created on Day 3 by task2_embeddings.py


def clean(text):
    text = text.lower()                        # lowercase
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove punctuation
    return " ".join(text.split())              # tokenise on spaces and re-join


def build_dataset():
    """2,000 short movie reviews (positive / negative) from the Rotten Tomatoes dataset."""
    from datasets import load_dataset
    raw = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
    df = raw.to_pandas().sample(n=2000, random_state=42)
    df["label"] = df["label"].map({0: "negative", 1: "positive"})
    df["text"] = df["text"].map(clean)
    df = df.drop_duplicates(subset="text")
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train.to_csv(BASE / "sentiment_train.csv", index=False)
    test.to_csv(BASE / "sentiment_test.csv", index=False)
    return train, test


def load_glove():
    data = np.load(GLOVE_CACHE)
    words = data["words"].tolist()
    return words, data["vectors"]


def main():
    train, test = build_dataset()
    print(f"Dataset: {len(train) + len(test)} texts | train {len(train)} | test {len(test)}")
    print("Label balance (train):", train["label"].value_counts().to_dict())
    print("Example:", train["text"].iloc[0][:100])

    # ---------------- 1. TF-IDF + Logistic Regression ----------------
    tfidf_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                LogisticRegression(max_iter=1000, C=5))
    tfidf_model.fit(train["text"], train["label"])
    pred = tfidf_model.predict(test["text"])
    tfidf_acc = accuracy_score(test["label"], pred)
    tfidf_f1 = f1_score(test["label"], pred, average="macro")
    print("\n" + "=" * 60 + "\nTF-IDF + LOGISTIC REGRESSION\n" + "=" * 60)
    print(f"Accuracy {tfidf_acc:.3f} | macro F1 {tfidf_f1:.3f}")
    labels = list(tfidf_model.classes_)
    cm = pd.DataFrame(confusion_matrix(test["label"], pred, labels=labels),
                      index=[f"true {l}" for l in labels], columns=labels)
    print("Confusion matrix:\n" + cm.to_string())

    # ---------------- 2. 10 most important words per class ----------------
    vec, clf = tfidf_model.named_steps["tfidfvectorizer"], tfidf_model.named_steps["logisticregression"]
    vocab = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]                       # binary: positive weight -> class 1, negative -> class 0
    print("\n10 most important words per class:")
    print(f"  {labels[1]:9}: {', '.join(vocab[np.argsort(coef)[::-1][:10]])}")
    print(f"  {labels[0]:9}: {', '.join(vocab[np.argsort(coef)[:10]])}")

    # ---------------- 3. GloVe word vectors ----------------
    words, vectors = load_glove()
    index = {w: i for i, w in enumerate(words)}
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def most_similar(positive, negative=(), topn=5):
        q = sum(unit[index[w]] for w in positive) - sum((unit[index[w]] for w in negative), np.zeros(50))
        sims = unit @ (q / np.linalg.norm(q))
        for w in list(positive) + list(negative):
            sims[index[w]] = -1
        return [(words[i], round(float(sims[i]), 3)) for i in np.argsort(-sims)[:topn]]

    print("\n" + "=" * 60 + "\nGLOVE (glove-wiki-gigaword-50)\n" + "=" * 60)
    for w in ["movie", "boring"]:
        print(f"5 nearest to '{w}': {most_similar([w])}")
    print(f"king - man + woman -> {most_similar(['king', 'woman'], ['man'], topn=1)[0][0]}")

    # ---------------- 4. Averaged sentence vectors as features ----------------
    def sentence_vector(text):
        vecs = [vectors[index[t]] for t in text.split() if t in index and t not in ENGLISH_STOP_WORDS]
        return np.mean(vecs, axis=0) if vecs else np.zeros(50)

    X_tr = np.vstack(train["text"].map(sentence_vector))
    X_te = np.vstack(test["text"].map(sentence_vector))
    glove_clf = LogisticRegression(max_iter=2000).fit(X_tr, train["label"])
    gpred = glove_clf.predict(X_te)
    glove_acc = accuracy_score(test["label"], gpred)
    glove_f1 = f1_score(test["label"], gpred, average="macro")

    print("\n" + "=" * 60 + "\nCOMPARISON (same test set)\n" + "=" * 60)
    print(f"TF-IDF features         : accuracy {tfidf_acc:.3f} | macro F1 {tfidf_f1:.3f}")
    print(f"Averaged GloVe features : accuracy {glove_acc:.3f} | macro F1 {glove_f1:.3f}")


if __name__ == "__main__":
    main()