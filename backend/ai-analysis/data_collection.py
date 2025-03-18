import os
import requests
import pandas as pd
import logging
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------
# Loglama Ayarları
# ---------------------------------------------------
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------
# Requests Oturumu ve Yeniden Deneme (Retry) Ayarları
# ---------------------------------------------------
def get_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = get_session()

# ---------------------------------------------------
# Büyük Veriyi Kaydetme Fonksiyonları
# ---------------------------------------------------
def save_to_parquet(df, filename):
    """Veriyi Parquet formatında sıkıştırarak kaydeder."""
    save_path = os.path.join(os.getcwd(), "data", f"{filename}.parquet")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ✅ Barkod sütununu string formatına çevir (ÖNEMLİ)
    if "barcode" in df.columns:
        df["barcode"] = df["barcode"].astype(str)

    df.to_parquet(save_path, index=False, engine="pyarrow")
    logging.debug(f"Data successfully saved in Parquet format: {save_path}")


def save_to_jsonl(df, filename):
    """Veriyi JSONL formatında kaydeder."""
    save_path = os.path.join(os.getcwd(), "data", f"{filename}.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_json(save_path, orient="records", lines=True, force_ascii=False)
    logging.debug(f"Data successfully saved in JSONL format: {save_path}")

# ---------------------------------------------------
# OpenFoodFacts API'den Veri Çekme ve İşleme
# ---------------------------------------------------
def fetch_openfoodfacts():
    logging.debug("Fetching OpenFoodFacts API data...")
    url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
    save_path = os.path.join(os.getcwd(), "data", "openfoodfacts_products.csv.gz")
    try:
        response = session.get(url, timeout=60, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        logging.debug(f"OpenFoodFacts dataset downloaded: {save_path}")
        return save_path
    except Exception as e:
        logging.error(f"Error fetching OpenFoodFacts data: {e}")
        return None

# ---------------------------------------------------
# OpenBeautyFacts API'den Veri Çekme ve İşleme
# ---------------------------------------------------
def fetch_openbeautyfacts():
    logging.debug("Fetching OpenBeautyFacts API data...")
    url = "https://static.openbeautyfacts.org/data/openbeautyfacts-products.jsonl.gz"
    save_path = os.path.join(os.getcwd(), "data", "openbeautyfacts_products.jsonl.gz")
    try:
        response = session.get(url, timeout=60, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        logging.debug(f"OpenBeautyFacts dataset downloaded: {save_path}")
        return save_path
    except Exception as e:
        logging.error(f"Error fetching OpenBeautyFacts data: {e}")
        return None

# ---------------------------------------------------
# Büyük CSV Verisini İşleme
# ---------------------------------------------------
def process_large_csv(gz_file, save_name, format_type="parquet", chunksize=100000):
    logging.debug(f"Processing CSV file: {gz_file}")
    df_list = []
    
    with gzip.open(gz_file, "rt", encoding="utf-8") as f:
        for i, chunk in enumerate(pd.read_csv(f, sep='\t', encoding="utf-8", chunksize=chunksize, on_bad_lines='skip')):
            logging.debug(f"Processing chunk {i+1}...")
            
            # ✅ Sadece gerekli sütunları al
            chunk = chunk[["code", "product_name", "brands", "ingredients_text"]].dropna()
            
            # ✅ Sütun adlarını düzenle
            chunk.rename(columns={"code": "barcode", "product_name": "name", "brands": "brand", "ingredients_text": "ingredients"}, inplace=True)
            
            # ✅ Barkod sütununu STRING olarak ZORUNLU hale getir
            chunk["barcode"] = chunk["barcode"].astype(str)
            
            df_list.append(chunk)

    df = pd.concat(df_list, ignore_index=True)

    # ✅ Parquet veya JSONL olarak kaydetme
    if format_type == "parquet":
        save_to_parquet(df, save_name)
    elif format_type == "jsonl":
        save_to_jsonl(df, save_name)


# ---------------------------------------------------
# Tüm Verilerin İşlenmesi ve Kaydedilmesi
# ---------------------------------------------------
def save_combined_data():
    logging.debug("Starting data collection...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(fetch_openfoodfacts): "openfoodfacts_data",
            executor.submit(fetch_openbeautyfacts): "openbeautyfacts_data"
        }
        results = {}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                logging.error(f"Error fetching {key}: {e}")
                results[key] = None
    if results.get("openfoodfacts_data"):
        process_large_csv(results["openfoodfacts_data"], "processed_openfoodfacts", format_type="parquet")
    if results.get("openbeautyfacts_data"):
        logging.debug("OpenBeautyFacts dataset downloaded and ready for further processing.")
    logging.debug("Data collection complete.")

if __name__ == "__main__":
    save_combined_data()
    logging.info("Data collection and processing completed.")
