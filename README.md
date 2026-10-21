# Customer Support Ticket Classifier (Weekly Project - Part 1)

Classify support tickets by **category** (queue) and **priority** (low / medium / high).
Branch: `feature/ticket-classifier`

## Dataset
- **Source:** Kaggle - *Customer IT Support - Ticket Dataset* by Tobias Bueck
  (file `dataset-tickets-multi-lang3-4k.csv`, saved here as `tickets_raw.csv`)
- Columns used: `subject` + `body` -> **text**, `queue` -> **category**, `priority` -> **priority**
- 4,000 tickets in 5 languages (en 1,391, de 848, es 812, fr 476, pt 473) - **only English kept: 1,391 tickets**

### Cleaning (`task1_prepare_data.py`)
- Joined subject + body, lowercased, trimmed extra spaces
- Removed empty tickets and duplicates (0 duplicates found)
- Merged queues with fewer than 30 tickets (General Inquiry, Human Resources) into **Other**

### Split (70 / 15 / 15, stratified)
| Split | Rows |
|---|---|
| train | 973 |
| validation | 209 |
| test | 209 |

Stratified on category (some category + priority combinations were too small to stratify on both).
Priority stayed balanced in every split: high ~43%, medium ~40%, low ~17-21%.

### Class balance (validation split, 209 tickets)
| Category | Count | | Priority | Count |
|---|---|---|---|---|
| Technical Support | 71 | | medium | 87 |
| Product Support | 36 | | high | 79 |
| Customer Service | 32 | | low | 43 |
| IT Support | 25 | | | |
| Billing and Payments | 18 | | | |
| Returns and Exchanges | 9 | | | |
| Service Outages and Maintenance | 7 | | | |
| Sales and Pre-Sales | 6 | | | |
| Other | 5 | | | |

The categories are **imbalanced** (Technical Support is ~34%, Other only ~2%), so **macro F1** is reported next to accuracy.

## Baseline: TF-IDF + Logistic Regression (`task2_baseline.py`)
- Two separate pipelines: one for **category**, one for **priority**
- `TfidfVectorizer(min_df=2, sublinear_tf=True)` + `LogisticRegression(class_weight="balanced")`
- Tuned `ngram_range` [(1,1), (1,2)] and `C` [0.1, 1, 10] with 5-fold GridSearchCV (scoring = macro F1)

| Label | Best params | CV macro F1 | Val accuracy | Val macro F1 |
|---|---|---|---|---|
| Category | C=10, ngram (1,1) | 0.484 | **0.507** | **0.566** |
| Priority | C=1, ngram (1,1) | 0.522 | **0.612** | **0.595** |

Models saved with joblib: `baseline_category.joblib`, `baseline_priority.joblib`.

### Priority confusion matrix (validation)
| true \ predicted | high | low | medium |
|---|---|---|---|
| high | 54 | 5 | 20 |
| low | 4 | 21 | 18 |
| medium | 18 | 16 | 53 |

### Error analysis - 10 most confusing tickets (and why)
| # | True -> Predicted | Ticket (start) | Likely reason |
|---|---|---|---|
| 1 | Other -> Technical Support | "outlook login failures through the organization's email portal..." | Login/email problem sounds technical; "Other" has very few examples |
| 2 | Returns and Exchanges -> Technical Support | "the keyboard on my dell xps 13 9310 has ceased to operate..." | Describes a hardware fault, not the return request |
| 3 | Sales and Pre-Sales -> Product Support | "issues with macbook air m1 purchased..." | Product name + "issues" look like support |
| 4 | Product Support -> Technical Support | "request for assistance with slack integration issues..." | Product vs Technical Support use the same words |
| 5 | IT Support -> Technical Support | "problem with norton antivirus plus subscription..." | IT and Technical Support overlap heavily |
| 6 | Customer Service -> Technical Support | "network assessment request... connectivity issues..." | Technical words inside a service request |
| 7 | Product Support -> Customer Service | "request for aws deployment guidance..." | Polite request style looks like Customer Service |
| 8 | Customer Service -> Technical Support | "urgent jira issue... critical issue with jira software 8.20..." | Software bug words, but routed to Customer Service |
| 9 | IT Support -> Customer Service | "hp deskjet 3755 connectivity issues dear tech online store customer support..." | "Customer support" greeting misleads the model |
| 10 | Technical Support -> Customer Service | "seeking wireless setup information for canon printer..." | Asking for information sounds like a service question |

### Observations
1. **Billing** is easy (16/18 correct) - unique words like invoice, charge, refund.
2. **Technical / Product / IT Support / Customer Service** are mixed up most - the same vocabulary appears in all four queues, and some labels look inconsistent.
3. Many tickets start with the same template ("I hope this message finds you well..."), which adds noise for TF-IDF.
4. **Priority** depends on urgency and business impact, not on topic words, so low vs medium is hard to separate.
5. Next step (Part 2): fine-tune DistilBERT, which reads context instead of single words, and compare with this baseline.