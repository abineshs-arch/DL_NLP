# Week 3 Day 4 (Part 1): Hugging Face - Tokenisers, Pipelines, BERT vs GPT

## Task 1: Tokenisers and Pipelines (`task1_pipelines.py`)

### AutoTokenizer (distilbert-base-uncased)
- Every sentence gets `[CLS]` at the start and `[SEP]` at the end
- Shorter sentences are padded with `[PAD]` (id 0) and the **attention_mask is 0** there, so the model ignores padding
- Example: "My payment failed twice." -> `[CLS] my payment failed twice . [SEP] [PAD] [PAD] [PAD]`
  - input_ids: `[101, 2026, 7909, 3478, 3807, 1012, 102, 0, 0, 0]`
  - attention_mask: `[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]`

**Sub-word tokens** (rare words are split into known pieces, `##` = continues previous piece):
| Word | Tokens |
|---|---|
| tokenisation | token, ##isation |
| unbelievably | un, ##bel, ##ie, ##va, ##bly |
| yoho | yo, ##ho |
| chennai | chennai (common enough to be one token) |

### Sentiment analysis pipeline
| Text | Label | Score |
|---|---|---|
| The support team solved my problem quickly, thank you! | POSITIVE | 1.000 |
| I have been waiting for a refund for two weeks. Terrible service. | NEGATIVE | 0.997 |
| The new update is fantastic. | POSITIVE | 1.000 |
| The app keeps crashing every time I open it. | NEGATIVE | 0.988 |
| Delivery was okay, nothing special. | NEGATIVE | 0.996 |

Note: the model (SST-2) only has POSITIVE/NEGATIVE, no NEUTRAL, so a neutral sentence is forced into one side.

### NER pipeline (dslim/bert-base-NER, aggregation_strategy="simple")
Text: "Ravi from Yoho Technologies paid Rs. 45000 in Chennai on 12 March"
| Entity | Type | Score |
|---|---|---|
| Ravi | PER (person) | 0.999 |
| Yoho Technologies | ORG (organisation) | 0.999 |
| Chennai | LOC (location) | 0.999 |

Note: "Rs. 45000" and "12 March" were not detected - this model only knows PER, ORG, LOC and MISC (no MONEY or DATE).

### Zero-shot classification (facebook/bart-large-mnli)
| Ticket | Prediction |
|---|---|
| I was charged twice for my subscription this month | **Billing** 0.86 |
| The app freezes when I upload a photo | **Technical** 0.92 |
| I forgot my password and cannot sign in | **Account** 0.91 |

With single-word labels ("Billing") the first ticket was wrongly predicted as Account (0.55).
Using descriptive labels ("billing, payments or charges") and the template
"This customer support ticket is about {}." fixed it.

## Task 2: BERT vs GPT (`task2_bert_gpt.py`)

### BERT fill-mask (bert-base-uncased)
| Sentence | Top 3 predictions |
|---|---|
| I was [MASK] twice for my subscription. | paid (0.08), arrested (0.07), asked (0.06) |
| Please reset my [MASK] because I cannot log in. | account (0.13), computer (0.12), password (0.11) |
| Chennai is a big city in [MASK]. | **india (0.98)**, kerala (0.01), chennai (0.01) |

### GPT-2 text generation (temperature 0.7, max_new_tokens 30)
- "Dear customer, thank you for contacting support. We" -> "...appreciate your patience and we will be happy to assist you in any way we can."
- "The best way to learn machine learning is" -> "...to look at the various types of data and how it relates to what you're learning."
- "Our new mobile app helps users" -> "...find the best places to eat and drink in their community."

### Parameter count
| Model | Parameters |
|---|---|
| BERT (bert-base-uncased) | 109,514,298 (~110M) |
| GPT-2 (gpt2) | 124,439,808 (~124M) |

### Comparison table
| | BERT | GPT |
|---|---|---|
| Architecture | Transformer **encoder** | Transformer **decoder** |
| Reads / writes | Reads the whole sentence in **both directions** at once | Reads **left to right** and writes the next word, one at a time |
| Training task | Fill in masked (hidden) words | Predict the next word |
| Best for | **Understanding** text | **Generating** text |
| Example tasks | Classification, NER, sentiment, search, fill-mask | Chatbots, drafting emails/replies, summaries, story writing |

### Why BERT for ticket classification and GPT for drafting replies
1. Ticket classification needs understanding of the whole ticket; BERT reads every word with context from both sides.
2. BERT outputs a label (Billing / Technical / Account), which is exactly what classification needs, and it is fast and cheap to fine-tune.
3. Drafting a reply needs new text to be written; GPT is trained to predict the next word, so it naturally generates fluent sentences.
4. So: BERT (encoder) to understand and route the ticket, GPT (decoder) to write the response.

---

# Week 3 Day 4 (Part 2): Fine-tuning with the Hugging Face Trainer API

## Task 1: Prepare data and Trainer (`task1_prepare.py`)
- Dataset: **dair-ai/emotion** (6 labels: sadness, joy, love, anger, fear, surprise)
- 2,500 random rows, stratified split: **2,000 train / 500 validation**
- Tokenised with `dataset.map(tokenize, batched=True)` (truncation, max_length 64)
- Model: `AutoModelForSequenceClassification` (distilbert-base-uncased), `num_labels=6`
- `DataCollatorWithPadding` pads each batch to the same length
- TrainingArguments: learning_rate 2e-5, 3 epochs, batch size 16, evaluation every epoch
- `compute_metrics`: accuracy and F1 (macro)
- Running the script prints the dataset size, label distribution and "TRAINER READY"

## Task 2: Fine-tune and evaluate (`task2_finetune.py`)
- `Trainer.train()` on CPU; metrics printed after each epoch
- Confusion matrix printed for the 500 validation rows
- Model and tokenizer saved with `save_pretrained("emotion_model")` and reloaded with
  `pipeline("text-classification", model="emotion_model")` to label 5 new sentences
  (e.g. "I am really nervous about my exam tomorrow." -> **fear** 0.705)

### Results (same 2,000 train / 500 validation split)
| Model | Accuracy | F1 (macro) |
|---|---|---|
| **Fine-tuned DistilBERT** | **0.784** | **0.640** |
| TF-IDF + Logistic Regression | 0.600 | 0.357 |

### Observations
1. Fine-tuning beats the TF-IDF baseline by ~18 accuracy points and almost doubles F1.
2. DistilBERT already understands language from pre-training; fine-tuning only teaches it the 6 emotion labels.
3. TF-IDF only sees which words appear, so it struggles when emotions are expressed with different words.
4. F1 is lower than accuracy because rare classes (love, surprise) have few examples; more data or class weights would help.

Note: `emotion_model/` (~270 MB) and `hf_output/` are in `.gitignore`, so they are not committed.
Run `task2_finetune.py` to recreate the model.