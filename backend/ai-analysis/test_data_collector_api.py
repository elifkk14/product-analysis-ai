import requests
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 📌 Loglama Ayarları
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

# 📌 Requests Oturumu ve Yeniden Deneme (Retry) Ayarları
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
# 📌 OpenFoodFacts API Testi (Genel Veri Çekme)
# ---------------------------------------------------
def test_fetch_food_data():
    logging.debug("Testing OpenFoodFacts API (General Data)...")
    url = "https://world.openfoodfacts.org/api/v0/search?fields=code,product_name,ingredients_text,brands"
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "products" in data:
            logging.info(f"✅ OpenFoodFacts API ÇALIŞIYOR - {len(data['products'])} ürün çekildi.")
        else:
            logging.warning("⚠️ OpenFoodFacts API'den beklenen veri formatı alınamadı.")
    except Exception as e:
        logging.error(f"❌ OpenFoodFacts API HATASI: {e}")

# ---------------------------------------------------
# 📌 OpenFoodFacts API Testi (Barkod ile Veri Çekme)
# ---------------------------------------------------
def test_fetch_food_by_barcode():
    barcode = "737628064502"
    logging.debug(f"Testing OpenFoodFacts API (Barcode: {barcode})...")
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "product" in data:
            logging.info(f"✅ Barkod API ÇALIŞIYOR - Ürün Adı: {data['product'].get('product_name', 'Unknown')}")
        else:
            logging.warning("⚠️ Barkod API'den beklenen veri formatı alınamadı.")
    except Exception as e:
        logging.error(f"❌ Barkod API HATASI: {e}")

# ---------------------------------------------------
# 📌 Zararlı Maddeler API Testi
# ---------------------------------------------------
def test_fetch_harmful_substances():
    logging.debug("Testing OpenFoodFacts Additives API...")
    url = "https://world.openfoodfacts.org/data/additives.json"
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "tags" in data:
            logging.info(f"✅ Zararlı Maddeler API ÇALIŞIYOR - {len(data['tags'])} madde bulundu.")
        else:
            logging.warning("⚠️ Zararlı Maddeler API'den beklenen veri formatı alınamadı.")
    except Exception as e:
        logging.error(f"❌ Zararlı Maddeler API HATASI: {e}")

# ---------------------------------------------------
# 📌 Tüm API Testlerini Çalıştır
# ---------------------------------------------------
if __name__ == "__main__":
    logging.info("🚀 API Testleri Başlatılıyor...")
    test_fetch_food_data()
    test_fetch_food_by_barcode()
    test_fetch_harmful_substances()
    logging.info("✅ Tüm API Testleri Tamamlandı.")
