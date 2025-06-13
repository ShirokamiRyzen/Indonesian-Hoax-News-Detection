import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

base_url = "https://turnbackhoax.id/page/"
max_pages = 200
data = []

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
}

# Buat folder 'data' kalau belum ada
os.makedirs("data", exist_ok=True)

def scrape_page(page):
    url = f"{base_url}{page}/"
    print(f"Mengambil halaman {page}...")
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
    except Exception as e:
        print(f"Gagal mengambil halaman {page}: {e}")
        return []  # Return empty list if failed

    soup = BeautifulSoup(res.text, "html.parser")
    articles = soup.select("article.mh-loop-item")

    for article in articles:
        title_tag = article.select_one("h3.entry-title a")
        date_tag = article.select_one("span.mh-meta-date")
        author_tag = article.select_one("span.mh-meta-author a")
        excerpt_tag = article.select_one("div.mh-excerpt p")

        data.append({
            "Judul": title_tag.text.strip() if title_tag else "",
            "Link": title_tag["href"] if title_tag else "",
            "Tanggal": date_tag.text.strip() if date_tag else "",
            "Author": author_tag.text.strip() if author_tag else "",
            "Isi Ringkas": excerpt_tag.text.strip() if excerpt_tag else ""
        })

for i in range(1, max_pages + 1):
    scrape_page(i)

# Simpan ke dalam folder 'data'
output_path = os.path.join("data", "turnbackhoax.csv")
df = pd.DataFrame(data)
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Data berhasil disimpan di {output_path} (total {len(df)} baris)")
