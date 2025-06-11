# Sistem Deteksi Hoax 🔍

Sistem otomatis untuk mendeteksi berita hoax menggunakan teknik Machine Learning dan Natural Language Processing (NLP) dalam bahasa Indonesia.

## 📋 Deskripsi

Proyek ini mengembangkan sistem yang dapat mengklasifikasikan artikel berita sebagai "HOAX" atau "VALID" berdasarkan konten teksnya. Sistem ini menggunakan data yang dikumpulkan dari sumber-sumber terpercaya seperti TurnBackHoax dan Kompas untuk melatih model machine learning.

## ✨ Fitur Utama

- **Web Scraping Otomatis**: Mengumpulkan data berita dari TurnBackHoax dan Kompas
- **Preprocessing Teks**: Pembersihan teks dengan stemming dan stopword removal
- **Multiple Models**: Implementasi Naive Bayes dan Random Forest
- **Evaluasi Model**: Analisis akurasi dan performa model
- **Prediksi Real-time**: Klasifikasi teks berita baru

## 🏗️ Struktur Proyek

```
Hoax-Detection/
├── scraper.py                 # Orchestrator untuk semua scraper
├── turnbackhoax_scraper.py    # Scraper untuk TurnBackHoax
├── kompas_scraper.py          # Scraper untuk Kompas
├── normalizer.py              # Preprocessing dan pembersihan teks
├── trainer.py                 # Training model machine learning
├── prediktor.py              # Prediksi berita baru
├── data/                     # Data mentah hasil scraping
│   ├── kompas.csv
│   └── turnbackhoax.csv
├── data_clean/               # Data yang sudah dibersihkan
│   ├── kompas_cleaned.csv
│   └── turnbackhoax_cleaned.csv
└── model/                    # Model yang sudah dilatih
    ├── nb.pkl               # Naive Bayes model
    ├── rf.pkl               # Random Forest model
    └── vectorizer.pkl       # TF-IDF Vectorizer
```

## 🚀 Cara Penggunaan

### 1. Instalasi Dependencies

```bash
pip install pandas scikit-learn nltk Sastrawi joblib beautifulsoup4 requests
```

### 2. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
```

### 3. Scraping Data

Jalankan scraper untuk mengumpulkan data dari kedua sumber:

```bash
python scraper.py
```

Atau jalankan scraper individual:

```bash
python turnbackhoax_scraper.py
python kompas_scraper.py
```

### 4. Preprocessing Data

Bersihkan dan normalkan teks:

```bash
python normalizer.py
```

### 5. Training Model

Latih model machine learning:

```bash
python trainer.py
```

Opsi tambahan:
```bash
python trainer.py --test-size 0.3 --model nb  # Hanya Naive Bayes
python trainer.py --model rf                   # Hanya Random Forest
```

### 6. Prediksi

Gunakan model untuk memprediksi teks baru:

```python
from prediktor import predict_text

# Prediksi single text
result = predict_text("Teks berita yang ingin diprediksi", model_type="nb")
print(f"Prediksi: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")

# Prediksi dari file CSV
from prediktor import predict_csv
predict_csv("input.csv", "output.csv", model_type="rf")
```

## 🔧 Konfigurasi

### Label Classification

Sistem menggunakan tag dalam judul untuk mengklasifikasikan berita:

**HOAX Tags:**
- `[SALAH]`
- `[PENIPUAN]` 
- `[FITNAH]`
- `[DISINFORMASI]`
- `[HOAX]`

**VALID Tags:**
- `[VALID]`
- `[BENAR]`
- `[FAKTA]`
- `[KLARIFIKASI]`

### Text Preprocessing

Pipeline preprocessing meliputi:
1. Konversi ke lowercase
2. Penghapusan angka
3. Penghapusan tanda baca
4. Penghapusan stopwords bahasa Indonesia
5. Stemming menggunakan Sastrawi

## 📊 Model Performance

Sistem menggunakan dua algoritma machine learning:

1. **Multinomial Naive Bayes**: Cocok untuk klasifikasi teks dengan asumsi independensi fitur
2. **Random Forest**: Ensemble method yang robust terhadap overfitting

Evaluasi model menggunakan:
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix

## 📁 Format Data

### Input Data (CSV)
```csv
Judul,Link,Tanggal,Author,Isi Ringkas
[VALID] Judul Berita,https://example.com,01/01/2025,Author,Konten berita...
```

### Output Prediction
```csv
Judul,Isi Ringkas,Prediction,Confidence_NB,Confidence_RF
Judul Berita,Konten...,VALID,0.85,0.92
```

## 🛠️ Requirements

- Python 3.8+
- pandas
- scikit-learn
- nltk
- Sastrawi
- joblib
- beautifulsoup4
- requests

## 📈 Roadmap

- [ ] Implementasi deep learning models (LSTM, BERT)
- [ ] Web interface untuk prediksi real-time
- [ ] API endpoints untuk integrasi eksternal
- [ ] Ekspansi sumber data
- [ ] Implementasi active learning

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:

1. Fork repository ini
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📝 Lisensi

Proyek ini menggunakan lisensi MIT. Lihat file `LICENSE` untuk detail lebih lanjut.

## 🙏 Acknowledgments

- [Sastrawi](https://github.com/sastrawi/sastrawi) untuk Indonesian stemming
- [NLTK](https://www.nltk.org/) untuk natural language processing tools
- [TurnBackHoax](https://turnbackhoax.id/) sebagai sumber data hoax
- [Kompas](https://www.kompas.com/) sebagai sumber berita valid

## 📞 Kontak

Jika ada pertanyaan atau saran, silakan buat issue di repository ini.

---

**⚠️ Disclaimer**: Sistem ini merupakan alat bantu dan tidak 100% akurat. Selalu lakukan verifikasi manual untuk keputusan penting.
