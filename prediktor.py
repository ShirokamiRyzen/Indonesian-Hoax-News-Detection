import os, sys, re, string, joblib
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------------------------------------
# Jalur file model
# -------------------------------------------------------------------------------------------------
MODEL_DIR   = "model"
VECT_PATH   = os.path.join(MODEL_DIR, "vectorizer.pkl")
NB_PATH     = os.path.join(MODEL_DIR, "nb.pkl")
RF_PATH     = os.path.join(MODEL_DIR, "rf.pkl")

# -------------------------------------------------------------------------------------------------
# Pre-processing utilitas
# -------------------------------------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
STEMMER  = StemmerFactory().create_stemmer()
STOPWORD = set(stopwords.words("indonesian"))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORD]
    return " ".join(STEMMER.stem(w) for w in tokens)

# -------------------------------------------------------------------------------------------------
# Muat model
# -------------------------------------------------------------------------------------------------
def load_models():
    missing = [p for p in (VECT_PATH, NB_PATH, RF_PATH) if not os.path.exists(p)]
    if missing:
        sys.exit("Model belum ditemukan. Jalankan trainer.py lebih dulu.")
    vectorizer = joblib.load(VECT_PATH)
    nb_model   = joblib.load(NB_PATH)
    rf_model   = joblib.load(RF_PATH)
    return vectorizer, nb_model, rf_model

VECT, NB, RF = load_models()
LABEL = {1: "HOAX", 0: "TIDAK HOAX"}

# -------------------------------------------------------------------------------------------------
# Hitung akurasi terkini (pada seluruh data_clean)
# -------------------------------------------------------------------------------------------------
def accuracy_on_corpus():
    frames = [
        pd.read_csv(os.path.join("data_clean", fn))
        for fn in os.listdir("data_clean")
        if fn.endswith("_cleaned.csv")
    ]
    if not frames:
        return None, None
    df = pd.concat(frames, ignore_index=True)
    tags = ("[SALAH]", "[PENIPUAN]", "[FITNAH]", "[DISINFORMASI]", "[HOAX]")
    y_true = df["Judul"].str.upper().apply(lambda j: 1 if any(t in j for t in tags) else 0)
    X = VECT.transform(df["Isi Ringkas Clean"].fillna(""))
    acc_nb = accuracy_score(y_true, NB.predict(X))
    acc_rf = accuracy_score(y_true, RF.predict(X))
    return acc_nb, acc_rf

ACC_NB, ACC_RF = accuracy_on_corpus()

# -------------------------------------------------------------------------------------------------
# Prediksi satu teks
# -------------------------------------------------------------------------------------------------
def predict(text: str):
    clean = clean_text(text)
    X = VECT.transform([clean])
    
    # Predict probabilities for Naïve Bayes and Random Forest
    prob_nb = NB.predict_proba(X)[0][1] * 100  # kelas 1 = HOAX
    prob_rf = RF.predict_proba(X)[0][1] * 100  # kelas 1 = HOAX
    
    # Get the label based on threshold
    lbl_nb = LABEL[int(prob_nb >= 50)]
    lbl_rf = LABEL[int(prob_rf >= 50)]
    
    return prob_nb, lbl_nb, prob_rf, lbl_rf

# -------------------------------------------------------------------------------------------------
# Antarmuka CLI
# -------------------------------------------------------------------------------------------------
def interactive():
    print("Mode interaktif — tekan Enter kosong untuk keluar.\n")
    while True:
        try:
            text = input("» ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        tampil(text)

def tampil(text: str):
    prob_nb, lbl_nb, prob_rf, lbl_rf = predict(text)
    
    # Display text
    print(f"\nTeks       : {text}")
    print(f"Naïve Bayes  : {lbl_nb} (prob HOAX {prob_nb:.1f}%)")
    print(f"Random Forest: {lbl_rf} (prob HOAX {prob_rf:.1f}%)")

    # Determine final classification
    if prob_nb > prob_rf:
        final_label = lbl_nb
        final_prob = prob_nb
    else:
        final_label = lbl_rf
        final_prob = prob_rf

    print(f"Final Prediction: {final_label} (prob HOAX {final_prob:.1f}%)")

    if ACC_NB is not None:
        print(f"Akurasi NB: {ACC_NB:.3f}   Akurasi RF: {ACC_RF:.3f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        tampil(" ".join(sys.argv[1:]))
    else:
        interactive()
