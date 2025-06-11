import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from typing import List, Dict

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36")
}

INDEX_URL = "https://indeks.kompas.com/"
PAGE_URL  = INDEX_URL + "?page={}"
MAX_PAGES = 50

# ──────────────────────────────────────────────────────────────
def scrape_index(pages: int = MAX_PAGES) -> List[Dict]:
    records: List[Dict] = []

    for page in range(1, pages + 1):
        url = INDEX_URL if page == 1 else PAGE_URL.format(page)
        print("Mengambil", url)
        try:
            html = requests.get(url, headers=HEADERS, timeout=30).text
        except Exception as e:
            print("Gagal mengambil", url, ":", e)
            continue

        soup = BeautifulSoup(html, "html.parser")

        # Pola 1 (desktop indeks)
        items = soup.select("div.article__list article")
        # Pola 2 (mobile/varian lain)
        if not items:
            items = soup.select("div.articleList.-list div.articleItem")

        for art in items:
            title_tag = art.select_one("h3.article__title a, h2.articleTitle")
            if not title_tag:
                continue
            link  = title_tag["href"] if title_tag.name == "a" else art.select_one("a.article-link")["href"]
            title = title_tag.get_text(strip=True)

            # Menambahkan [VALID] di depan judul
            title = "[VALID] " + title

            # Tanggal / penulis (bila ada)
            date_tag   = art.select_one("div.article__date, div.articlePost-date")
            author_tag = art.select_one("div.article__author")
            tanggal = date_tag.get_text(strip=True) if date_tag else ""
            author  = author_tag.get_text(strip=True) if author_tag else ""

            excerpt = get_excerpt(link)
            records.append({
                "Judul": title,
                "Link": link,
                "Tanggal": tanggal,
                "Author": author,
                "Isi Ringkas": excerpt
            })
    return records

# ──────────────────────────────────────────────────────────────
def get_excerpt(url: str) -> str:
    try:
        res = requests.get(url, headers=HEADERS, timeout=30)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        p = soup.select_one("div.read__content p")
        return p.get_text(strip=True) if p else ""
    except Exception:
        return ""

# ──────────────────────────────────────────────────────────────
def save_csv(rows: List[Dict], fname: str = "kompas.csv"):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", fname)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Data tersimpan di {path} (total {len(rows)} baris)")

# ──────────────────────────────────────────────────────────────
def main():
    data = scrape_index()
    save_csv(data)

if __name__ == "__main__":
    main()
