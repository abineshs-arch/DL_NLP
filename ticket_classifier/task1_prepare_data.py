"""Task 1: load the Kaggle ticket dataset, keep English, clean, check balance, split 70/15/15."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
MIN_PER_CATEGORY = 30          # queues smaller than this are merged into "Other"


def clean(text):
    return " ".join(str(text).lower().split())      # lowercase + trim + single spaces


def strat_key(df):
    """category_priority; rare combinations fall back to just the category."""
    key = df["category"] + "_" + df["priority"]
    rare = key.map(key.value_counts()) < 10
    return key.where(~rare, df["category"])


def main():
    raw = pd.read_csv(BASE / "tickets_raw.csv")
    print(f"Raw rows (all languages): {len(raw)}")
    print("Languages:", raw["language"].value_counts().to_dict())

    df = raw[raw["language"] == "en"].copy()
    print(f"English rows: {len(df)}")

    # text = subject + body ; category = queue ; priority = priority
    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).map(clean)
    df["category"] = df["queue"].astype(str).str.strip()
    df["priority"] = df["priority"].astype(str).str.strip().str.lower()
    df = df[["text", "category", "priority"]]

    df = df[df["text"].str.len() > 10]
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate/empty tickets -> {len(df)} rows")

    small = df["category"].value_counts()
    small = small[small < MIN_PER_CATEGORY].index
    if len(small):
        print(f"Merged small queues into 'Other': {list(small)}")
        df.loc[df["category"].isin(small), "category"] = "Other"

    for col in ["category", "priority"]:
        counts = df[col].value_counts()
        pct = (counts / len(df) * 100).round(1)
        print(f"\n{col.upper()} balance:")
        print(pd.DataFrame({"count": counts, "percent": pct}).to_string())

    # 70 / 15 / 15, stratified on category + priority so BOTH labels keep their balance
    df["strat"] = strat_key(df)
    try:
        train, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["strat"])
        val, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["strat"])
        print("\nStratified on category + priority")
    except ValueError:
        train, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["category"])
        val, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["category"])
        print("\nStratified on category (some category+priority groups were too small)")

    print(f"Split -> train {len(train)} | validation {len(val)} | test {len(test)}")
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        part.drop(columns="strat").to_csv(BASE / f"{name}.csv", index=False)
        pri = (part["priority"].value_counts(normalize=True) * 100).round(0).astype(int).to_dict()
        print(f"  {name:10} priority %: {pri}")
    print("\nSaved train.csv, validation.csv, test.csv")


if __name__ == "__main__":
    main()