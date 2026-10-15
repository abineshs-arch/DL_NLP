import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

tickets = [
    "I was charged twice for my subscription",
    "The app crashes when I open settings",
    "Please reset my password, cannot log in",
    "Refund not received for cancelled order",
    "Error 500 when uploading a file",
    "How do I change my registered email",
]


def preprocess(text):
    text = text.lower()                          # 1. lowercase
    text = re.sub(r"[^a-z0-9\s]", " ", text)     # 2. remove punctuation
    return text.split()                          # 3. tokenise (split on spaces)


def main():
    print("Preprocessing:")
    for t in tickets:
        print(f"  {t!r:<45} -> {preprocess(t)}")

    # ---- TF-IDF on the 6 tickets ----
    clean = [" ".join(preprocess(t)) for t in tickets]
    vec = TfidfVectorizer()
    X = vec.fit_transform(clean)
    words = vec.get_feature_names_out()
    print(f"\nTF-IDF matrix shape: {X.shape}  (6 tickets x {X.shape[1]} unique words)")

    row = X[0].toarray().ravel()
    top = row.argsort()[::-1][:5]
    print("\nTop 5 words in ticket 1:")
    for i in top:
        print(f"  {words[i]:<14} {row[i]:.3f}")

    # ---- Most similar ticket to a new query ----
    query = "cannot log in to my account"
    q = vec.transform([" ".join(preprocess(query))])
    sims = cosine_similarity(q, X).ravel()
    print(f"\nQuery: {query!r}")
    for i in sims.argsort()[::-1]:
        print(f"  ticket {i + 1}: similarity {sims[i]:.3f} | {tickets[i]}")
    best = sims.argmax()
    print(f"-> Most similar: ticket {best + 1}: {tickets[best]!r}")

    # ---- Text classification on 30 labelled tickets ----
    labelled = [
        ("I was charged twice for my subscription", "billing"),
        ("Refund not received for cancelled order", "billing"),
        ("My invoice shows the wrong amount", "billing"),
        ("Please cancel my subscription and refund me", "billing"),
        ("Payment failed but money was deducted", "billing"),
        ("Why was my card charged again this month", "billing"),
        ("I need a copy of my last invoice", "billing"),
        ("The discount code was not applied to my payment", "billing"),
        ("Double payment taken from my bank account", "billing"),
        ("How long does a refund take", "billing"),
        ("The app crashes when I open settings", "technical"),
        ("Error 500 when uploading a file", "technical"),
        ("The page is very slow and keeps loading", "technical"),
        ("App freezes after the latest update", "technical"),
        ("Images are not uploading, upload fails", "technical"),
        ("The website shows a blank screen", "technical"),
        ("Notifications are not working on my phone", "technical"),
        ("Error message when I click save", "technical"),
        ("The app crashes on startup", "technical"),
        ("Download button does nothing", "technical"),
        ("Please reset my password, cannot log in", "account"),
        ("How do I change my registered email", "account"),
        ("I forgot my password", "account"),
        ("My account is locked after too many attempts", "account"),
        ("How do I delete my account", "account"),
        ("Cannot log in with my username", "account"),
        ("I want to change my phone number on the account", "account"),
        ("Two factor login code not received", "account"),
        ("Update the name on my profile", "account"),
        ("Login link in the email has expired", "account"),
    ]
    texts = [" ".join(preprocess(t)) for t, _ in labelled]
    labels = [lab for _, lab in labelled]

    clf = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.3,
                                              random_state=42, stratify=labels)
    clf.fit(X_tr, y_tr)
    print(f"\n30 labelled tickets (10 billing / 10 technical / 10 account)")
    print(f"Test accuracy (9 held-out tickets): {clf.score(X_te, y_te):.3f}")
    cv = cross_val_score(make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)),
                         texts, labels, cv=5)
    print(f"5-fold CV accuracy: {cv.mean():.3f} +/- {cv.std():.3f}")

    clf.fit(texts, labels)
    for new in ["I was billed two times", "the app keeps crashing", "reset my login password"]:
        print(f"  {new!r:<28} -> {clf.predict([' '.join(preprocess(new))])[0]}")


if __name__ == "__main__":
    main()