# Customer Support Ticket Classifier

Classifies a support ticket into a **category** (queue) and a **priority** (low / medium / high)
and serves the model through a **FastAPI** endpoint.
Branch: `feature/ticket-classifier` - all project files are in `ticket_classifier/`.

| File | Purpose |
|---|---|
| `task1_prepare_data.py` | Clean data, class balance, 70/15/15 stratified split |
| `task2_baseline.py` | TF-IDF + Logistic Regression baselines, tuning, error analysis |
| `task3_finetune_compare.py` | Fine-tune DistilBERT and compare with the baseline |
| `app.py` | FastAPI service: `POST /predict`, `GET /health` |
| `test_app.py` | 6 pytest tests with TestClient |

---

## 1. Dataset
- **Source:** Kaggle - *Customer IT Support - Ticket Dataset* by Tobias Bueck
  (`dataset-tickets-multi-lang3-4k.csv`, saved as `tickets_raw.csv`)
- `subject` + `body` -> **text**, `queue` -> **category**, `priority` -> **priority**
- 4,000 tickets in 5 languages (en 1,391, de 848, es 812, fr 476, pt 473) - **English only: 1,391**
- Cleaning: lowercase, trim spaces, remove empty/duplicate tickets (0 duplicates);
  queues with < 30 tickets (General Inquiry, Human Resources) merged into **Other**
- Split 70/15/15 stratified on category: **train 973 | validation 209 | test 209**
  (priority stayed balanced: high ~43%, medium ~40%, low ~17-21%)

Class balance (validation, 209 tickets): Technical Support 71, Product Support 36, Customer Service 32,
IT Support 25, Billing and Payments 18, Returns and Exchanges 9, Service Outages 7, Sales 6, Other 5
- **imbalanced**, so macro F1 is reported next to accuracy.

## 2. Baseline: TF-IDF + Logistic Regression
- `TfidfVectorizer(min_df=2, sublinear_tf=True)` + `LogisticRegression(class_weight="balanced")`
- 5-fold GridSearchCV over `ngram_range` [(1,1),(1,2)] and `C` [0.1, 1, 10], scoring = macro F1

| Label | Best params | Val accuracy | Val macro F1 |
|---|---|---|---|
| Category | C=10, ngram (1,1) | 0.507 | 0.566 |
| Priority | C=1, ngram (1,1) | 0.612 | 0.595 |

**Error analysis (10 most confusing tickets):** most errors are between Technical Support, Product Support,
IT Support and Customer Service - they use the same vocabulary ("issue", "software", "connectivity") and some
labels look inconsistent. Many tickets share the template "I hope this message finds you well...", which adds noise.
Billing is easy (16/18 correct) because of unique words (invoice, charge, refund).
Priority depends on urgency, not topic words, so low vs medium is hard.

## 3. Fine-tuned DistilBERT vs baseline (category)
DistilBERT fine-tuned with the Trainer API (lr 5e-5, 3 epochs, batch 16, max_length 128, CPU).
Validation F1 per epoch: 0.195 -> 0.311 -> 0.353 (still improving = under-trained).

| Model | Split | Accuracy | Macro F1 | Training time | Inference / ticket |
|---|---|---|---|---|---|
| TF-IDF + LogReg | validation | **0.507** | **0.566** | **0.4 s** | **0.09 ms** |
| TF-IDF + LogReg | test | **0.493** | **0.470** | **0.4 s** | **0.09 ms** |
| DistilBERT | validation | 0.502 | 0.353 | 910 s | 78 ms |
| DistilBERT | test | 0.474 | 0.306 | 910 s | 77 ms |

- **DistilBERT wins** on Technical Support (F1 0.616 vs 0.545), the largest class.
- It scores **0** on small classes (IT Support, Sales, Other) - with 973 tickets and 3 CPU epochs it never learned them.
- **The baseline is good enough** for clear classes: Billing 0.84, Returns 0.70.

### Model chosen for deployment: TF-IDF + Logistic Regression
1. **Better quality now:** test macro F1 0.470 vs 0.306, similar accuracy.
2. **Speed:** ~0.09 ms per ticket vs ~77 ms - about 870x faster.
3. **Cost:** trains in < 1 s, a few MB model, runs on a cheap CPU server; DistilBERT needs ~15 min and a ~270 MB model.
4. **Fairness to small queues:** the baseline still predicts IT Support, Sales and Other.
5. **Revisit later:** with more data, a GPU and more epochs DistilBERT should overtake it - re-run the comparison then.

## 4. API (`app.py`)
- Models are loaded **once at startup** (FastAPI `lifespan`), not on every request
- Input validated with **Pydantic**: empty or whitespace-only text returns **422**

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | - | `{"status": "ok", "models_loaded": true}` |
| POST | `/predict` | `{"text": "..."}` | category, priority, confidence, priority_confidence |

### Run steps (Windows, from the `DL_NLP` folder)
```powershell
# 1. install
python -m pip install fastapi uvicorn httpx pytest scikit-learn pandas joblib

# 2. (only if the .joblib files are missing) rebuild data and models
python ticket_classifier/task1_prepare_data.py
python ticket_classifier/task2_baseline.py

# 3. start the API
python -m uvicorn app:app --app-dir ticket_classifier

# 4. open the interactive docs
#    http://127.0.0.1:8000/docs
```

### Example requests
```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for my subscription, please refund"}'
```
Response:
```json
{"category": "Billing and Payments", "priority": "Medium", "confidence": 0.821, "priority_confidence": 0.463}
```
PowerShell alternative:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body '{"text": "I was charged twice"}'
```

### Tests
```powershell
python -m pytest ticket_classifier -v
```
6 passed: health, valid prediction, empty text 422, spaces-only 422, missing field 422, wrong type 422.

## 5. Limitations
- Category accuracy is ~50%: similar support queues overlap heavily and labels are noisy.
- Trained only on 1,391 English tickets; other languages are not supported.
- `confidence` is the model's probability, not a guarantee - low-confidence tickets should go to a human.
- `distilbert_category/` (~270 MB) is not committed; run `task3_finetune_compare.py` to recreate it.