import os
import json
import time
import logging
import asyncio
import aiohttp
import pandas as pd
from tqdm import tqdm
from jsonschema import validate
from async_lru import alru_cache
import re
from langdetect import detect, LangDetectException  # Dil tespiti için

# ------------------------------
# 🔧 Yapılandırma
# ------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class TrainingDataConfig:
    # Giriş verisi dosya yolu (hem food hem kozmetik kayıtlarını içeriyor)
    INPUT_PATH = os.path.join(SCRIPT_DIR, "data", "processed_openfoodfacts.parquet")
    # Çıktı dosya yolu (CSV olarak kaydetmek için)
    OUTPUT_PATH = "optimized_training_data.csv"
    # Toplam alınacak örnek kayıt sayısı (örneğin 1000, dengeli olmak üzere 500 gıda, 500 kozmetik)
    TOTAL_SAMPLE_SIZE = 1000
    # İşlemde eş zamanlı kullanılacak maksimum işçi sayısı
    MAX_WORKERS = 3
    # Her batch’de işlenecek kayıt sayısı
    BATCH_SIZE = 10
    # AI model API bilgileri
    API_ENDPOINT = "http://localhost:11434/api/generate"
    MODEL_NAME = "mistral:7b-instruct-q4_K_M"
    TIMEOUT = 90
    CACHE_LIMIT = 600
    MAX_RETRIES = 3
    MODEL_PARAMS = {
        "temperature": 0.3,
        "num_ctx": 4096,
        "repeat_penalty": 1.5
    }
    # Çıktı formatı: "csv" (parquet için ekleme yapmak daha karmaşık olduğu için CSV önerilir)
    SAVE_FORMAT = "csv"

# ------------------------------
# 📊 Veri Temizleme ve Örnekleme
# ------------------------------
class DataCleaner:
    @staticmethod
    def load_data() -> pd.DataFrame:
        logging.info(f"Veri dosyası okunuyor: {TrainingDataConfig.INPUT_PATH}")
        df = pd.read_parquet(TrainingDataConfig.INPUT_PATH)
        return df

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["ingredients"])
        df = df.dropna(subset=["ingredients"])
        df = df[df["ingredients"].str.len() > 10]
        logging.info(f"Temizleme sonrası toplam kayıt sayısı: {len(df)}")
        return df

    @staticmethod
    def balance_data(df: pd.DataFrame) -> pd.DataFrame:
        if "category" in df.columns:
            food_df = df[df["category"].str.contains("food", case=False, na=False)]
            cosmetic_df = df[df["category"].str.contains("cosmetic", case=False, na=False)]
        else:
            food_df = df
            cosmetic_df = pd.DataFrame(columns=df.columns)

        half_sample = TrainingDataConfig.TOTAL_SAMPLE_SIZE // 2
        food_sample = food_df.sample(n=min(half_sample, len(food_df)), random_state=42) if not food_df.empty else pd.DataFrame()
        cosmetic_sample = cosmetic_df.sample(n=min(half_sample, len(cosmetic_df)), random_state=42) if not cosmetic_df.empty else pd.DataFrame()

        balanced_df = pd.concat([food_sample, cosmetic_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
        logging.info(f"Dengeli örnekleme sonrası kayıt sayısı: {len(balanced_df)}")
        return balanced_df

    @staticmethod
    def is_english(text: str) -> bool:
        try:
            return detect(text) == "en"
        except LangDetectException:
            return False

    @staticmethod
    def translate_to_english(text: str) -> str:
        if DataCleaner.is_english(text):
            return text
        else:
            # İsteğe bağlı: Çeviri API entegrasyonu yapılabilir.
            return text  # veya boş string döndürebilirsiniz

    @staticmethod
    def preprocess_data() -> pd.DataFrame:
        df = DataCleaner.load_data()
        df = DataCleaner.clean_data(df)
        df = DataCleaner.balance_data(df)
        df["ingredients"] = df["ingredients"].apply(lambda x: DataCleaner.translate_to_english(x))
        df = df[df["ingredients"].str.strip() != ""]
        return df[["ingredients"]]

# ------------------------------
# 🤖 AI Analiz ve Veri Zenginleştirme
# ------------------------------
class AIAnnotator:
    SCHEMA = {
        "type": "object",
        "properties": {
            "safe_ingredients": {"type": "array"},
            "harmful_ingredients": {"type": "array"},
            "general_health_assessment": {"type": "string"}
        },
        "required": ["safe_ingredients", "harmful_ingredients", "general_health_assessment"]
    }

    def __init__(self):
        self.semaphore = asyncio.Semaphore(TrainingDataConfig.MAX_WORKERS * 2)
        self.start_time = time.time()

    def validate_response(self, response: dict) -> bool:
        try:
            validate(instance=response, schema=self.SCHEMA)
            return True
        except Exception as e:
            logging.error(f"Şema doğrulama hatası: {str(e)}")
            return False

    import re

    def fix_response_keys(self, response: dict) -> dict:
        normalized = {}
        for key, value in response.items():
            norm_key = re.sub(r'[\s_]', '', key).lower()
            if "harmful" in norm_key:
                normalized.setdefault("harmful_ingredients", [])
                if isinstance(value, list):
                    normalized["harmful_ingredients"].extend(value)
                else:
                    normalized["harmful_ingredients"].append(value)
            elif "safe" in norm_key:
                normalized.setdefault("safe_ingredients", [])
                if isinstance(value, list):
                    normalized["safe_ingredients"].extend(value)
                else:
                    normalized["safe_ingredients"].append(value)
            elif "generalhealthassess" in norm_key:
                if isinstance(value, list):
                    normalized["general_health_assessment"] = " ".join(str(item) for item in value)
                elif isinstance(value, dict):
                    normalized["general_health_assessment"] = str(next(iter(value.values())))
                else:
                    normalized["general_health_assessment"] = str(value)
            else:
                pass

        if "safe_ingredients" not in normalized:
            normalized["safe_ingredients"] = []
        if "harmful_ingredients" not in normalized:
            normalized["harmful_ingredients"] = []
        if "general_health_assessment" not in normalized:
            normalized["general_health_assessment"] = ""

        # Tüm öğeleri string'e dönüştürüyoruz
        normalized["safe_ingredients"] = list(dict.fromkeys([str(x) for x in normalized["safe_ingredients"]]))
        normalized["harmful_ingredients"] = list(dict.fromkeys([str(x) for x in normalized["harmful_ingredients"]]))

        return normalized

    @alru_cache(maxsize=TrainingDataConfig.CACHE_LIMIT)
    async def analyze_ingredients(self, session: aiohttp.ClientSession, ingredients: str) -> dict:
        prompt = f"""
        **STRICT JSON RESPONSE REQUIRED**
        You are a food safety expert. Analyze the following product ingredients meticulously.
        For each ingredient, decide whether it is safe or potentially harmful.
        If an ingredient could be harmful, list it in the "harmful_ingredients" array.
        If none are harmful, return an empty array for "harmful_ingredients".
        List all remaining ingredients in the "safe_ingredients" array.
        Provide a brief overall health assessment in "general_health_assessment".

        Return the answer in the exact JSON format below:
        {{
            "safe_ingredients": [list safe ingredients],
            "harmful_ingredients": [list harmful ingredients],
            "general_health_assessment": "A brief overall health assessment based on the ingredients."
        }}

        Ingredients: {ingredients}
        """
        try:
            async with session.post(
                TrainingDataConfig.API_ENDPOINT,
                json={
                    "model": TrainingDataConfig.MODEL_NAME,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": TrainingDataConfig.MODEL_PARAMS
                },
                timeout=TrainingDataConfig.TIMEOUT
            ) as response:
                data = await response.json()
                raw_json = data.get("response", "").strip()
                logging.info(f"Raw response: {raw_json}")

                start_index = raw_json.find('{')
                end_index = raw_json.rfind('}') + 1
                clean_json = raw_json[start_index:end_index]
                parsed = json.loads(clean_json)
                parsed = self.fix_response_keys(parsed)
                logging.info(f"Düzeltilmiş yanıt: {json.dumps(parsed, ensure_ascii=False)}")

                if not self.validate_response(parsed):
                    raise ValueError("Üretilen JSON şeması geçerli değil")
                return parsed
        except Exception as e:
            logging.error(f"Analiz hatası: {str(e)}")
            return {
                "safe_ingredients": [],
                "harmful_ingredients": [],
                "general_health_assessment": f"Error: {str(e)}"
            }

    async def process_batch(self, session: aiohttp.ClientSession, batch: pd.DataFrame) -> list:
        async with self.semaphore:
            tasks = [self.analyze_ingredients(session, row["ingredients"]) for _, row in batch.iterrows()]
            return await asyncio.gather(*tasks, return_exceptions=False)

# ------------------------------
# 🚀 Ana Fonksiyon: Eğitim Verisini Hazırlama
# ------------------------------
async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logging.info("Eğitim veri dosyası hazırlama işlemi başlatılıyor...")

    df = DataCleaner.preprocess_data()
    total_records = len(df)
    logging.info(f"Ön işleme sonrası toplam kayıt: {total_records}")

    annotator = AIAnnotator()
    total_batches = (total_records + TrainingDataConfig.BATCH_SIZE - 1) // TrainingDataConfig.BATCH_SIZE

    # Eğer çıktı dosyası varsa, silerek başlıyoruz (veya devam etmek isterseniz resume mekanizması ekleyebilirsiniz)
    if os.path.exists(TrainingDataConfig.OUTPUT_PATH):
        os.remove(TrainingDataConfig.OUTPUT_PATH)

    async with aiohttp.ClientSession() as session:
        for i in range(0, total_records, TrainingDataConfig.BATCH_SIZE):
            batch = df.iloc[i:i+TrainingDataConfig.BATCH_SIZE]
            for attempt in range(TrainingDataConfig.MAX_RETRIES):
                try:
                    batch_result = await annotator.process_batch(session, batch)
                    break
                except Exception as e:
                    logging.error(f"Batch {i // TrainingDataConfig.BATCH_SIZE} hatası: {str(e)}")
                    if attempt == TrainingDataConfig.MAX_RETRIES - 1:
                        batch_result = [{
                            "safe_ingredients": [],
                            "harmful_ingredients": [],
                            "general_health_assessment": f"Batch error: {str(e)}"
                        }] * len(batch)
                    else:
                        await asyncio.sleep(2 ** attempt)

            # Batch sonuçlarını DataFrame'e çevirip orijinal batch ile birleştiriyoruz
            batch_df = pd.DataFrame(batch_result)
            combined = pd.concat([batch.reset_index(drop=True), batch_df], axis=1)

            # Incremental olarak CSV'ye ekliyoruz; dosya yoksa header yazılır.
            if TrainingDataConfig.SAVE_FORMAT.lower() == "csv":
                combined.to_csv(
                    TrainingDataConfig.OUTPUT_PATH,
                    mode='a',
                    index=False,
                    header=not os.path.exists(TrainingDataConfig.OUTPUT_PATH)
                )
            else:
                # Parquet incremental ekleme daha karmaşık olduğundan, burayı ayrı ele alabilirsiniz.
                pass

            # Batch ilerlemesini logluyoruz
            processed_batches = (i // TrainingDataConfig.BATCH_SIZE) + 1
            elapsed = time.time() - annotator.start_time
            avg_time = elapsed / processed_batches
            remaining = total_batches - processed_batches
            logging.info(f"İşlenen batch: {processed_batches}/{total_batches} - Tahmini kalan süre: {avg_time * remaining:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
