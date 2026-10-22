# Week 3 Assessment - Deep Learning + NLP

All code is in this folder. Run each script from the `DL_NLP` folder.

| Task | File |
|---|---|
| 1. Neural network fundamentals | `task1_fundamentals.py` -> `xor_loss_curves.png` |
| 2. PyTorch training pipeline | `task2_pytorch_pipeline.py` -> `mlp_best.pt`, `mlp_training.png` |
| 3. Classical NLP baseline + GloVe | `task3_nlp_baseline.py` -> `sentiment_train.csv`, `sentiment_test.csv` |
| 4. Transformers + FastAPI | `task4_transformers.py`, `api.py`, `test_api.py` |

---

## Task 1 - Neural network fundamentals (NumPy)
- `relu`, `sigmoid`, `softmax`, `mse`, `cross_entropy` written in NumPy
- `softmax([2.0, 1.0, 0.1]) = [0.659 0.242 0.099]`
- Gradient of `(w*x + b - t)^2` at w=1, x=2, b=0, t=10:
  error = 2 - 10 = -8 -> dL/dw = 2·error·x = **-32**, dL/db = 2·error = **-16**
  PyTorch autograd: `w.grad = -32.0, b.grad = -16.0` (matches)
- 2-4-1 network (ReLU hidden, Sigmoid output), binary cross-entropy, gradient descent, 5000 epochs

| Learning rate | Loss @100 | Loss @1000 | Final loss | XOR predictions |
|---|---|---|---|---|
| **0.5** | 0.058 | 0.001 | 0.0002 | **[0, 1, 1, 0]** |
| 0.01 (too slow) | 0.666 | 0.506 | 0.058 | [0.11, 0.96, 0.96, 0.04] |
| 5 (unstable) | 0.693 | 0.693 | 0.693 | [0.5, 0.5, 0.5, 0.5] |

With lr=5 the first updates are so large that the ReLU units die and the network gets stuck predicting 0.5.

### Regularisation in 2 lines each
- **Dropout:** randomly switches off a fraction of neurons during training, so the network cannot rely on any single neuron.
  This reduces overfitting; at test time all neurons are used.
- **BatchNorm:** normalises each layer's outputs over the mini-batch to mean 0 / std 1 (then learns a scale and shift).
  This keeps activations stable, allows higher learning rates and speeds up training.
- **Early stopping:** watch the validation metric every epoch and stop when it has not improved for N epochs (patience).
  Keep the weights from the best epoch, so the model does not overfit by training too long.

## Task 2 - PyTorch training pipeline (breast cancer data)
- 569 rows, 30 features, split 455 train / 114 validation (stratified), `StandardScaler` fitted on train only
- `CsvDataset(Dataset)` with `__len__` / `__getitem__`; DataLoaders with batch_size=32 (shuffle for train)
- `MLP(nn.Module)`: Linear(30,64) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(64,32) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(32,1)
- **Params: 4,289**
- Training loop: `zero_grad -> forward -> loss -> backward -> step`, Adam (lr 1e-3), BCEWithLogitsLoss, up to 40 epochs, early stopping (patience 8)

| Result | Value |
|---|---|
| Best val accuracy | **0.9737 at epoch 3** (early stopping at epoch 11) |
| Reloaded model (new MLP + `load_state_dict`, eval mode) | **0.9737 - identical** |
| scikit-learn LogisticRegression (same split) | 0.9825 |

On this small, clean, almost linearly separable dataset, logistic regression is as good as (slightly better than) the neural network.

## Task 3 - Classical NLP baseline and embeddings
- Dataset: **2,000 short reviews** (Rotten Tomatoes, positive / negative), split 1,600 train / 400 test (stratified)
- Cleaning: lowercase, remove punctuation, tokenise on spaces
- TF-IDF (1-2 grams) + Logistic Regression: **accuracy 0.700, macro F1 0.699**

| | pred negative | pred positive |
|---|---|---|
| true negative | 127 | 70 |
| true positive | 50 | 153 |

**10 most important words:**
- positive: and, performance, refreshing, may, performances, good, psychological, history, of, beautifully
- negative: too, dull, bad, any, over, like, nothing, video, off, unfunny

**GloVe (glove-wiki-gigaword-50):**
- nearest to *movie*: movies, film, films, comedy, hollywood
- nearest to *boring*: awfully, pretty, funny, incredibly, silly
- **king - man + woman -> queen**

| Features (Logistic Regression) | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF | **0.700** | **0.699** |
| Averaged GloVe sentence vectors | 0.688 | 0.687 |

Averaging word vectors loses word order and negation, and GloVe puts opposites close together
("boring" is near "funny" because they appear in similar contexts), so TF-IDF is slightly better here.

## Task 4 - Transformers, fine-tuning and serving
- `task4_transformers.py` prints the DistilBERT tokenizer output (tokens, input_ids, attention_mask) for 2 sentences
  and runs the `sentiment-analysis` and `ner` pipelines on 3 sentences each
- Fine-tuned `distilbert-base-uncased` with the Trainer API: lr 3e-5, 3 epochs, batch 16, max_length 64, CPU

### Comparison table (same 400 test reviews)
| Model | Accuracy | Macro F1 | Training time | Inference time |
|---|---|---|---|---|
| TF-IDF + LogReg | 0.700 | 0.699 | **0.2 s** | **0.03 ms / text** |
| **DistilBERT (fine-tuned)** | **0.800** | **0.800** | 562 s (~9 min) | 28 ms / text |

DistilBERT is **10 points better** because sentiment depends on context and negation ("not bad", "too long"),
which a transformer reads but TF-IDF only counts as separate words. The cost is ~900x slower inference and a ~270 MB model.

### API (`api.py`)
- `POST /predict` with `{"text": "..."}` returns `label` and `confidence`
- The model is loaded **once at startup** (FastAPI `lifespan`), not per request
- Pydantic validation: empty or whitespace-only text returns **422**
- `GET /health` returns `{"status": "ok", "model_loaded": true}`
- `test_api.py`: **5 tests passed** (health, valid prediction, empty 422, spaces 422, missing field 422)

Example:
```
POST /predict {"text": "The service was slow and the staff were rude"}
-> 200 {"label": "negative", "confidence": 0.979}
```

Run:
```powershell
python week3_assessment/task4_transformers.py          # creates week3_assessment/sentiment_model/
python -m pytest week3_assessment -v
python -m uvicorn api:app --app-dir week3_assessment    # then open http://127.0.0.1:8000/docs
```
`sentiment_model/` (~270 MB) is in `.gitignore`; run `task4_transformers.py` to recreate it.

### BERT (encoder) vs GPT (decoder) - which for this task
1. BERT is an encoder: it reads the whole sentence in both directions at once and builds a meaning vector for it.
2. GPT is a decoder: it reads left to right and is trained to predict the next word, so it is built to generate text.
3. Classifying feedback is an understanding task with a fixed set of labels, not a writing task.
4. A fine-tuned BERT-style model (DistilBERT) gives a label and a confidence quickly and cheaply; here it reached 0.80 vs 0.70 for TF-IDF.
5. So I would use **DistilBERT (encoder)** for classifying feedback, and a GPT-style model only if we later need to write replies.