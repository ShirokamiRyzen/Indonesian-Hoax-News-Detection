import os
import argparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

HOAX_TAGS     = ("[SALAH]", "[PENIPUAN]", "[FITNAH]", "[DISINFORMASI]", "[HOAX]")
NON_HOAX_TAGS = ("[VALID]", "[BENAR]", "[FAKTA]", "[KLARIFIKASI]")

def make_label(title: str) -> int | None:
    title = title.upper()

    # Cek untuk HOAX
    if any(t in title for t in HOAX_TAGS):
        return 1
    # Cek untuk VALID
    if any(t in title for t in NON_HOAX_TAGS):
        return 0
    # Untuk Kompas, jika tidak ada tag, kita anggap valid
    if "kompas" in title:
        return 0  # Label VALID
    # Jika tak jelas, dibuang
    return None

def load_data(clean_dir: str = "data_clean"):
    frames = [
        pd.read_csv(os.path.join(clean_dir, fn))
        for fn in os.listdir(clean_dir)
        if fn.endswith("_cleaned.csv")
    ]
    if not frames:
        raise FileNotFoundError("Tidak ada *_cleaned.csv di data_clean/")

    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["Judul"].apply(make_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    texts  = df["Isi Ringkas Clean"].fillna("")
    labels = df["label"]

    print("Distribusi kelas:")
    print(labels.value_counts().rename({0: "VALID", 1: "HOAX"}), "\n")

    # Jika ada hanya satu kelas, tampilkan peringatan dan hentikan proses
    if len(labels.value_counts()) == 1:
        raise ValueError("Hanya ada satu kelas (VALID/HOAX) di dataset. Model tidak dapat dilatih.")

    return texts, labels

def train_models(texts, labels, ngram_low: int, ngram_high: int):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(ngram_low, ngram_high),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    nb = MultinomialNB()
    nb.fit(X_tr, y_tr)

    rf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=42
    )
    rf.fit(X_tr, y_tr)

    print("\n===== HASIL EVALUASI =====")
    for name, model in [("Naïve Bayes", nb), ("Random Forest", rf)]:
        print(f"\n{name}")
        y_pred = model.predict(X_te)
        print(f"Akurasi : {accuracy_score(y_te, y_pred):.3f}")
        print(classification_report(y_te, y_pred, target_names=["VALID", "HOAX"], zero_division=0))
        print("Confusion matrix [tn fp; fn tp]:\n", confusion_matrix(y_te, y_pred))

    return vectorizer, nb, rf

def save_models(vec, nb, rf, model_dir: str = "model"):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(vec, os.path.join(model_dir, "vectorizer.pkl"))
    joblib.dump(nb,  os.path.join(model_dir, "nb.pkl"))
    joblib.dump(rf,  os.path.join(model_dir, "rf.pkl"))
    print(f"\nModel dan vectorizer disimpan di '{model_dir}/'")

def main():
    parser = argparse.ArgumentParser(description="Latih model deteksi hoax / valid")
    parser.add_argument(
        "--ngram",
        nargs=2,
        type=int,
        metavar=("LOW", "HIGH"),
        default=[1, 2],
        help="rentang n-gram TF-IDF (default 1 2)",
    )
    args = parser.parse_args()
    n_low, n_high = args.ngram

    print(f"Memuat data, n-gram {n_low}–{n_high}")
    texts, labels = load_data()
    vec, nb, rf = train_models(texts, labels, n_low, n_high)
    save_models(vec, nb, rf)

if __name__ == "__main__":
    main()
