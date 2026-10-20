"""
download_scf.py — Downloads the Survey of Consumer Finances (SCF) 2022 dataset.

The SCF is conducted by the Federal Reserve every 3 years. It contains
detailed data on household finances, including income, debt, assets, and
net worth. Used here to benchmark financial health clustering models.

Reference: Federal Reserve Board, Survey of Consumer Finances (2022),
https://www.federalreserve.gov/econres/scfindex.htm
"""

import io
import os
import shutil
import zipfile

import pandas as pd
import requests

# SCF 2022 public dataset URL (CSV format, released by the Federal Reserve)
SCF_URL = "https://www.federalreserve.gov/econres/files/scfp2022excel.zip"

RAW_DIR = "MLAI/clustering/data/raw"
OUTPUT_PATH = os.path.join(RAW_DIR, "scf2022.csv")


def download_scf():
    os.makedirs(RAW_DIR, exist_ok=True)

    print(f"Downloading SCF 2022 from:\n  {SCF_URL}\n")
    response = requests.get(SCF_URL, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunks = []

    for chunk in response.iter_content(chunk_size=8192):
        chunks.append(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            print(f"\r  {pct:.1f}% ({downloaded:,} / {total:,} bytes)", end="", flush=True)

    print("\nDownload complete. Extracting zip...")

    zip_bytes = b"".join(chunks)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        print(f"  Files in zip: {names}")

        # Find the CSV file inside the zip (e.g. SCFP2022.csv)
        csv_files = [n for n in names if n.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError("No CSV file found in the zip archive.")

        csv_name = csv_files[0]
        print(f"  Extracting: {csv_name} -> {OUTPUT_PATH}")

        with zf.open(csv_name) as src, open(OUTPUT_PATH, "wb") as dst:
            shutil.copyfileobj(src, dst)

    df = pd.read_csv(OUTPUT_PATH)

    print("\nDone.")
    print(f"  Rows   : {len(df):,}")
    print(f"  Columns: {len(df.columns):,}")
    print(f"  Column names (first 20): {list(df.columns[:20])}")


if __name__ == "__main__":
    download_scf()
