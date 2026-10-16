import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import pipeline, set_seed


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def bert_fill_mask():
    print("=" * 60)
    print("PART A: BERT (bert-base-uncased) - fill-mask")
    print("=" * 60)
    fill = pipeline("fill-mask", model="bert-base-uncased")
    sentences = [
        "I was [MASK] twice for my subscription.",
        "Please reset my [MASK] because I cannot log in.",
        "Chennai is a big city in [MASK].",
    ]
    for s in sentences:
        preds = fill(s, top_k=3)
        guesses = ", ".join(f"{p['token_str']} ({p['score']:.2f})" for p in preds)
        print(f"\n  {s}\n    -> {guesses}")
    return count_params(fill.model)


def gpt2_generate():
    print("\n" + "=" * 60)
    print("PART B: GPT-2 - text generation (temperature 0.7, 30 new tokens)")
    print("=" * 60)
    gen = pipeline("text-generation", model="gpt2")
    set_seed(42)   # same output every run
    prompts = [
        "Dear customer, thank you for contacting support. We",
        "The best way to learn machine learning is",
        "Our new mobile app helps users",
    ]
    for p in prompts:
        out = gen(p, max_new_tokens=30, do_sample=True, temperature=0.7,
                  pad_token_id=gen.tokenizer.eos_token_id)
        print(f"\n  PROMPT: {p}\n  OUTPUT: {out[0]['generated_text']}")
    return count_params(gen.model)


def main():
    bert_params = bert_fill_mask()
    gpt_params = gpt2_generate()

    print("\n" + "=" * 60)
    print("PART C: Parameter count")
    print("=" * 60)
    print(f"  BERT (bert-base-uncased): {bert_params:,}  (~{bert_params / 1e6:.0f}M)")
    print(f"  GPT-2 (gpt2)            : {gpt_params:,}  (~{gpt_params / 1e6:.0f}M)")


if __name__ == "__main__":
    main()