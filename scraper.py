import importlib
import runpy
import sys

SCRAPER_MODULES = (
    "turnbackhoax_scraper",
    "kompas_scraper",
)

def run_module(mod_name: str):
    """Jalankan modul dengan prioritas:
       1) fungsi main() bila ada
       2) run_as_script via runpy
    """
    try:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "main"):
            mod.main()
        else:
            # Eksekusi seperti python -m modul
            runpy.run_module(mod_name, run_name="__main__")
    except Exception as exc:
        print(f"Gagal menjalankan {mod_name}: {exc}", file=sys.stderr)

def main():
    for mod in SCRAPER_MODULES:
        print(f"\n==== Menjalankan {mod} ====")
        run_module(mod)

if __name__ == "__main__":
    main()