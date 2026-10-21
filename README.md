---

# Part 2 - Task 1: Fine-tuned DistilBERT vs Baseline (`task3_finetune_compare.py`)

DistilBERT (distilbert-base-uncased) fine-tuned for **category** with the Trainer API
(lr 5e-5, 3 epochs, batch 16, max_length 128, CPU). Priority keeps the TF-IDF baseline.

### DistilBERT per epoch (validation)
| Epoch | Val loss | Accuracy | Macro F1 |
|---|---|---|---|
| 1 | 1.561 | 0.411 | 0.195 |
| 2 | 1.398 | 0.483 | 0.311 |
| 3 | 1.355 | 0.502 | 0.353 |

### Comparison table (category)
| Model | Split | Accuracy | Macro F1 | Training time | Inference time / ticket |
|---|---|---|---|---|---|
| TF-IDF + LogReg | validation | **0.507** | **0.566** | **0.4 s** | **0.09 ms** |
| TF-IDF + LogReg | test | **0.493** | **0.470** | **0.4 s** | **0.09 ms** |
| DistilBERT (fine-tuned) | validation | 0.502 | 0.353 | 910 s (~15 min) | 78 ms |
| DistilBERT (fine-tuned) | test | 0.474 | 0.306 | 910 s (~15 min) | 77 ms |

### Per-category F1 on test - where each model wins
| Category | TF-IDF | DistilBERT | Winner |
|---|---|---|---|
| Billing and Payments | 0.842 | 0.850 | tie |
| Customer Service | 0.190 | 0.145 | TF-IDF |
| IT Support | 0.464 | 0.000 | TF-IDF |
| Other | 0.000 | 0.000 | tie |
| Product Support | 0.469 | 0.449 | tie |
| Returns and Exchanges | 0.700 | 0.471 | TF-IDF |
| Sales and Pre-Sales | 0.400 | 0.000 | TF-IDF |
| Service Outages and Maintenance | 0.615 | 0.222 | TF-IDF |
| Technical Support | 0.545 | **0.616** | **DistilBERT** |

### Analysis
- **DistilBERT wins** only on the largest class (Technical Support) - it learns the big classes first.
- It scores **0** on small classes (IT Support, Sales, Other): with 973 tickets and 3 CPU epochs it never learned them.
  F1 was still rising every epoch (0.195 -> 0.311 -> 0.353), so it is **under-trained**, not a bad model.
- **The baseline is good enough** for clear-vocabulary classes (Billing 0.84, Returns 0.70) and is much fairer
  to small classes thanks to `class_weight="balanced"`.

### Model chosen for deployment: TF-IDF + Logistic Regression
1. **Better quality now:** test macro F1 0.470 vs 0.306, and similar accuracy (0.493 vs 0.474).
2. **Speed:** ~0.09 ms per ticket vs ~77 ms - about 870x faster, so it can handle many tickets per second on a cheap CPU server.
3. **Cost:** trains in under 1 second and the model files are a few MB, vs ~15 min training and a ~270 MB model; no GPU needed.
4. **Fairness to small queues:** DistilBERT ignores IT Support, Sales and Other; the baseline still predicts them.
5. **When to switch:** with more labelled tickets, a GPU and more epochs, DistilBERT should overtake the baseline - re-run this comparison then.