import os
import re
import string
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
from multiprocessing import Pool

# Download stopwords (sekali saja)
nltk.download('stopwords', quiet=True)

# Inisialisasi stemmer dan stopwords
stemmer = StemmerFactory().create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# Fungsi untuk membersihkan teks
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))  # Hapus tanda baca
    tokens = [w for w in text.split() if w not in stop_words]  # Hapus stopwords
    return ' '.join(stemmer.stem(w) for w in tokens)  # Stemming

# Fungsi untuk memproses batch data
def process_batch(df_chunk):
    df_chunk["Isi Ringkas Clean"] = df_chunk["Isi Ringkas"].apply(clean_text)
    return df_chunk

# Path folder input dan output
input_folder, output_folder = "data", "data_clean"
os.makedirs(output_folder, exist_ok=True)

# Fungsi untuk memproses seluruh data dengan multiprocessing
def process_data_in_parallel():
    all_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    # Bagi data dalam chunks
    for fn in all_files:
        print(f"Membersihkan {fn}...")
        df = pd.read_csv(os.path.join(input_folder, fn))

        # Pastikan kolom yang diperlukan ada (periksa Isi Ringkas atau lainnya)
        if "Isi Ringkas" not in df.columns:
            print(f"Kolom 'Isi Ringkas' tidak ditemukan di {fn}")
            continue

        # Bagi data menjadi beberapa chunk untuk paralelisasi
        num_chunks = 4  # Anda bisa menyesuaikan jumlah chunks dengan jumlah core CPU
        chunk_size = len(df) // num_chunks
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        # Gunakan multiprocessing Pool untuk proses paralel
        with Pool(processes=num_chunks) as pool:
            result = pool.map(process_batch, chunks)

        # Gabungkan hasilnya kembali
        df_cleaned = pd.concat(result, ignore_index=True)

        # Simpan file yang sudah dibersihkan
        out_path = os.path.join(output_folder, fn.replace(".csv", "_cleaned.csv"))
        df_cleaned.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Disimpan di: {out_path}")

if __name__ == "__main__":
    process_data_in_parallel()
