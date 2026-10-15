# Week 3 Day 3 (Part 2): NLP Basics

## Task 1: Preprocessing and TF-IDF (`task1_tfidf.py`)
- Preprocessing: lowercase, remove punctuation, tokenise (split into words)
- TF-IDF matrix shape: **(6, 32)** (6 tickets, 32 unique words)
- Top words in ticket 1: was, twice, subscription, charged (0.441), for (0.361)
- **Similarity result:** query "cannot log in to my account" is most similar to
  **ticket 3: "Please reset my password, cannot log in"** (cosine similarity 0.733)
- Classifier (TF-IDF + LogisticRegression, 30 tickets, 3 classes):
  - Test accuracy: 0.556 (only 9 test tickets, so not reliable)
  - 5-fold cross-validation accuracy: **0.833 ± 0.105** (more reliable)

### What TF-IDF does
1. **TF (term frequency):** how often a word appears in this document.
2. **IDF (inverse document frequency):** words that appear in many documents (like "the") get a low weight; rare words get a high weight.
3. **TF × IDF:** a word scores high when it is frequent in this document but rare in others, so it describes this document well.

## Task 2: Word embeddings (`task2_embeddings.py`)
Used pre-trained **GloVe glove-wiki-gigaword-50** (400,000 words, 50 numbers each).
gensim does not support Python 3.14 yet, so the same gensim-data file is loaded with NumPy.

- 5 nearest to **king**: prince, queen, ii, emperor, son
- 5 nearest to **computer**: computers, software, technology, electronic, internet
- **king − man + woman → queen** (0.852)

| Sentence pair | Embeddings | TF-IDF |
|---|---|---|
| "My payment failed" vs "The transaction was declined" | **0.700** | 0.000 |
| "My payment failed" vs "The screen went black" | 0.483 | 0.000 |

### Why embeddings beat TF-IDF
1. TF-IDF only matches exact words, so "payment" and "transaction" look completely unrelated (similarity 0).
2. Embeddings place words with similar meanings close together, so synonyms get high similarity.
3. Embeddings are small and dense (50 numbers), while TF-IDF vectors are huge and mostly zeros.
4. Embeddings capture relationships, e.g. king − man + woman ≈ queen.

## Bonus: PCA plot
`glove_pca.png` shows 30 word vectors in 2D. Similar words (royalty, tech, cities, animals, food) form groups.