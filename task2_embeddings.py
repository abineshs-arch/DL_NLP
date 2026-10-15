import gzip
import os
import re
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
       "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz")
GZ_FILE = "glove-wiki-gigaword-50.gz"
CACHE = "glove50.npz"


def load_glove():
    """Same GloVe vectors that gensim.downloader uses, loaded with NumPy."""
    if os.path.exists(CACHE):
        data = np.load(CACHE)
        return data["words"].tolist(), data["vectors"]
    if not os.path.exists(GZ_FILE):
        print("Downloading GloVe (66 MB, one time only)...")
        urllib.request.urlretrieve(URL, GZ_FILE)
    print("Reading vectors (about 30 s, one time only)...")
    words, vecs = [], []
    with gzip.open(GZ_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != 51:          # skip the header line "400000 50"
                continue
            words.append(parts[0])
            vecs.append(np.asarray(parts[1:], dtype=np.float32))
    vectors = np.vstack(vecs)
    np.savez(CACHE, words=np.array(words), vectors=vectors)
    return words, vectors


words, vectors = load_glove()
index = {w: i for i, w in enumerate(words)}
unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)   # for cosine similarity


def most_similar(positive, negative=(), topn=5):
    """Like gensim's most_similar: add 'positive' vectors, subtract 'negative' ones."""
    query = sum(unit[index[w]] for w in positive) - sum((unit[index[w]] for w in negative), np.zeros(50))
    query /= np.linalg.norm(query)
    sims = unit @ query
    for w in list(positive) + list(negative):     # don't return the input words
        sims[index[w]] = -1
    best = np.argsort(-sims)[:topn]
    return [(words[i], round(float(sims[i]), 3)) for i in best]


def sentence_vector(sentence):
    tokens = re.sub(r"[^a-z0-9\s]", " ", sentence.lower()).split()
    known = [vectors[index[t]] for t in tokens
             if t in index and t not in ENGLISH_STOP_WORDS]   # skip "i", "my", "to"...
    return np.mean(known, axis=0)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print(f"Loaded {len(words):,} words, each a {vectors.shape[1]}-number vector\n")

    for w in ["king", "computer"]:
        print(f"5 nearest words to '{w}': {most_similar([w])}")

    print(f"\nking - man + woman -> {most_similar(['king', 'woman'], ['man'], topn=3)}")

    s1 = "My payment failed"
    s2 = "The transaction was declined"     # same meaning, NO shared words
    s3 = "The screen went black"            # different topic
    v1, v2, v3 = sentence_vector(s1), sentence_vector(s2), sentence_vector(s3)
    tf = TfidfVectorizer().fit_transform([s1, s2, s3])
    tf_sim = cosine_similarity(tf)
    print("\nSentence similarity:            embeddings | TF-IDF")
    print(f"  '{s1}' vs '{s2}':  {cosine(v1, v2):.3f}   |  {tf_sim[0, 1]:.3f}")
    print(f"  '{s1}' vs '{s3}':         {cosine(v1, v3):.3f}   |  {tf_sim[0, 2]:.3f}")

    # Bonus: 30 word vectors in 2D with PCA
    plot_words = ["king", "queen", "man", "woman", "prince", "princess",
                  "computer", "software", "laptop", "internet", "keyboard", "data",
                  "india", "chennai", "delhi", "mumbai", "london", "paris",
                  "dog", "cat", "horse", "cow", "lion", "tiger",
                  "apple", "banana", "mango", "rice", "bread", "milk"]
    pts = PCA(n_components=2).fit_transform(vectors[[index[w] for w in plot_words]])
    plt.figure(figsize=(10, 7))
    plt.scatter(pts[:, 0], pts[:, 1])
    for w, (x, y) in zip(plot_words, pts):
        plt.annotate(w, (x, y), xytext=(4, 4), textcoords="offset points")
    plt.title("30 GloVe word vectors projected to 2D with PCA")
    plt.grid(alpha=0.3)
    plt.savefig("glove_pca.png", dpi=150, bbox_inches="tight")
    print("\nSaved glove_pca.png")


if __name__ == "__main__":
    main()