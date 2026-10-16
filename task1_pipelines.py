import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # hide a harmless Windows warning

from transformers import AutoTokenizer, pipeline


def tokenizer_demo():
    print("=" * 60)
    print("PART A: AutoTokenizer (distilbert-base-uncased)")
    print("=" * 60)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    sentences = [
        "I cannot log in to my account.",
        "My payment failed twice.",
        "The app crashes.",
    ]
    batch = tok(sentences, padding=True)   # padding makes all rows the same length

    for i, s in enumerate(sentences):
        ids = batch["input_ids"][i]
        print(f"\nSentence {i + 1}: {s}")
        print("  tokens        :", tok.convert_ids_to_tokens(ids))
        print("  input_ids     :", ids)
        print("  attention_mask:", batch["attention_mask"][i])

    print("\nRare words are split into sub-word pieces (## = continues previous piece):")
    for w in ["tokenisation", "unbelievably", "chennai", "yoho"]:
        print(f"  {w:13} -> {tok.tokenize(w)}")


def sentiment_demo():
    print("\n" + "=" * 60)
    print("PART B: Sentiment analysis pipeline")
    print("=" * 60)
    clf = pipeline("sentiment-analysis",
                   model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    texts = [
        "The support team solved my problem quickly, thank you!",
        "I have been waiting for a refund for two weeks. Terrible service.",
        "The new update is fantastic.",
        "The app keeps crashing every time I open it.",
        "Delivery was okay, nothing special.",
    ]
    for text, r in zip(texts, clf(texts)):
        print(f"  {r['label']:8} {r['score']:.3f}  | {text}")


def ner_demo():
    print("\n" + "=" * 60)
    print("PART C: Named Entity Recognition (NER) pipeline")
    print("=" * 60)
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    text = "Ravi from Yoho Technologies paid Rs. 45000 in Chennai on 12 March"
    print(f"Text: {text}\n")
    for e in ner(text):
        print(f"  {e['entity_group']:5} {e['word']:20} score={e['score']:.3f}")


def zero_shot_demo():
    print("\n" + "=" * 60)
    print("PART D: Zero-shot classification pipeline")
    print("=" * 60)
    zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    # Descriptive labels work better than single words; map them back to short names
    labels = {
        "billing, payments or charges": "Billing",
        "a technical problem or bug in the app": "Technical",
        "account access, login or password": "Account",
    }
    tickets = [
        "I was charged twice for my subscription this month",
        "The app freezes when I upload a photo",
        "I forgot my password and cannot sign in",
    ]
    for t in tickets:
        r = zsc(t, candidate_labels=list(labels),
                hypothesis_template="This customer support ticket is about {}.")
        scores = ", ".join(f"{labels[l]} {s:.2f}" for l, s in zip(r["labels"], r["scores"]))
        print(f"  {labels[r['labels'][0]]:9} <- '{t}'   ({scores})")


def main():
    tokenizer_demo()
    sentiment_demo()
    ner_demo()
    zero_shot_demo()


if __name__ == "__main__":
    main()