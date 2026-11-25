import os
import re
import json
import time

import requests
import ollama
import pandas as pd
import logging
import psutil
from typing import List, Dict, Any
from pydantic import BaseModel, field_validator, ValidationError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor

# ------------------------------
# Configuration
# ------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")

class Config:
    INPUT_PATH = os.getenv("INPUT_PATH", "./data/")
    OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./results/")
    RISK_WEIGHTS = {
        "High": 4,
        "Medium": 2,
        "Low": 1,
        "Carcinogenic": 3,
        "Endocrine Disruptor": 2,
        "Allergen": 1
    }
    @classmethod
    def validate_paths(cls):
        """Dizinleri oluştur ve izinleri kontrol et"""
        try:
            # Giriş dizini kontrolü
            if not os.path.exists(cls.INPUT_PATH):
                logging.warning(f"⚠️ Input dizini yok: {cls.INPUT_PATH}")
                os.makedirs(cls.INPUT_PATH, exist_ok=True)
                
            # Çıkış dizini oluşturma
            os.makedirs(cls.OUTPUT_PATH, exist_ok=True)
            
            # Yazma izni kontrolü
            if not os.access(cls.OUTPUT_PATH, os.W_OK):
                raise PermissionError(f"✖️ Yazma izni yok: {cls.OUTPUT_PATH}")
                
            logging.info(f"📂 Dizinler hazır: INPUT={cls.INPUT_PATH} OUTPUT={cls.OUTPUT_PATH}")
            
        except Exception as e:
            logging.critical(f"❌ Dizin hatası: {str(e)}")
            raise

    @classmethod
    def get_batch_size(cls) -> int:
        """Dinamik batch boyutu belirleme"""
        free_mem = psutil.virtual_memory().available // (1024 ** 3)  # GB cinsinden
        return max(1, free_mem // 2)

    @classmethod
    def get_max_workers(cls) -> int:
        """Sistem kaynaklarına göre thread sayısı"""
        return min(4, (os.cpu_count() or 1) // 2)

    @classmethod
    def load_substances(cls) -> Dict[str, Any]:
        """Madde verilerini yükleme"""
        base_path = os.getenv("DATA_PATH", "./data/")
        
        try:
            df_harmful = pd.read_parquet(os.path.join(base_path, "harmful_substances.parquet"))
            df_safe = pd.read_parquet(os.path.join(base_path, "safe_substances.parquet"))
            
            df_harmful = df_harmful.drop_duplicates(subset=["substance"])
            df_safe["description"] = df_safe.get("description", "No description available")
            
            return {
                'harmful': df_harmful.set_index("substance")[["category", "reason", "severity"]].to_dict(orient='index'),
                'safe': df_safe.set_index("substance")["description"].to_dict()
            }
        except Exception as e:
            logging.critical(f"Substance data loading failed: {str(e)}")
            raise

# ------------------------------
# Ollama Integration
# ------------------------------
class SafetyAnalyzer:
    @staticmethod
    def analyze_ingredients(ingredients: List[str]) -> Dict[str, Any]:
        """Ollama API ile içerik analizi yapar ve yanıtı doğrular."""
        try:
            # Net talimatlarla İngilizce prompt
            response = ollama.generate(
                model="mistral:latest",
                prompt=f"""SYSTEM ROLE: You are a chemical safety expert. Return STRICT JSON ONLY.

                ANALYSIS REQUEST: Analyze these cosmetic ingredients for health risks:
                {json.dumps(ingredients)}

                RESPONSE FORMAT:
                {{
                "harmful_ingredients": [{{"name": "chemical_name", "description": "health risk reason"}}],
                "safe_ingredients": ["safe_ingredient_name"],
                "description": "max 50 word summary"
                }}

                INSTRUCTIONS:
                1. Analyze ONLY the provided ingredients
                2. Never add explanations
                3. Use scientific terminology
                4. Output MUST be valid JSON without formatting issues""",
                                stream=False,
                                format="json"
                            )

            raw_response = response.get('response', '').strip()
            logging.info(f"📄 Raw Response:\n{raw_response}")

            # Geliştirilmiş JSON çıkarımı
            json_match = re.search(
                r"(?:```json\s*)?({.*})\s*(?:```)?",
                raw_response,
                re.DOTALL
            )
            
            if json_match:
                clean_json = json_match.group(1)
                # JSON temizleme
                clean_json = re.sub(r'[\x00-\x1F]+', '', clean_json)  # Control karakterleri kaldır
                parsed_response = json.loads(clean_json)
            else:
                try:
                    parsed_response = json.loads(raw_response)
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON format"}

            # Katı validasyon
            if not SafetyAnalyzer._validate_response(parsed_response):
                logging.error("🚨 Validation failed for response")
                return {"error": "Response validation failed"}

            return parsed_response

        except Exception as e:
            logging.error(f"❌ Critical Error: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def _validate_response(response: Dict) -> bool:
        """Yanıtın yapısını kontrol eder"""
        required_keys = {
            "harmful_ingredients": list,
            "safe_ingredients": list,
            "description": str
        }
        
        # Anahtar varlık kontrolü
        if not all(key in response for key in required_keys):
            logging.error(f"Missing keys: {required_keys - response.keys()}")
            return False
            
        # Veri tipi kontrolü
        type_checks = [
            (isinstance(response['harmful_ingredients'], list)),
            (isinstance(response['safe_ingredients'], list)),
            (isinstance(response['description'], str))
        ]
        if not all(type_checks):
            logging.error("Type mismatch in response")
            return False
            
        # Zararlı içeriklerin yapısı
        for item in response['harmful_ingredients']:
            if not all(k in item for k in ("name", "description")):
                logging.error("Invalid harmful_ingredients structure")
                return False
                
        return True

    @staticmethod
    def _parse_response(raw_text: str, attempt: int) -> Dict[str, Any]:
        """Yanıt JSON formatında değilse, ham metinden JSON çıkarmaya çalışır."""
        try:
            # JSON kod bloğu varsa temizle
            json_match = re.search(r"(?:```json\s*)?(\{.*\})\s*(?:```)?", raw_text, re.DOTALL)
            clean_json = json_match.group(1) if json_match else raw_text

            # JSON olarak parse et
            parsed_response = json.loads(clean_json)

            # JSON'da anahtarların olup olmadığını kontrol et
            required_keys = {"harmful_ingredients", "safe_ingredients", "description"}
            missing_keys = required_keys - parsed_response.keys()

            if missing_keys:
                logging.warning(f"⚠️ JSON yanıtında eksik anahtarlar var: {missing_keys}")

            return parsed_response

        except (AttributeError, json.JSONDecodeError, ValueError) as e:
            logging.warning(f"⚠️ Yanıt JSON formatında değil, fallback stratejisi uygulanıyor. Hata: {e}")

            # Fallback: Önemli kelimeleri yakalayan bir regex ile yanıt üret
            harmful = list(set(re.findall(
                r'\b(paraben|sulfate|phthalate|fragrance|triclosan)\b',
                raw_text,
                flags=re.IGNORECASE
            )))

            return {
                "harmful_ingredients": [{"name": item, "description": "Detected as potentially harmful"} for item in harmful],
                "safe_ingredients": [],
                "description": "Automatic keyword-based risk analysis",
            }

# ------------------------------
# Data Processing
# ------------------------------
class DataProcessor:
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Güvenli metin temizleme"""
        return re.sub(r"[^a-zA-Z0-9\s,.-]", "", str(text))[:500]

    @staticmethod
    def parse_ingredients(text: str) -> List[str]:
        """Daha kapsamlı içerik ayrıştırma"""
        text = re.sub(r"(?i)\b(and|or|with|contains?|ingredients?)\b", ",", text)
        return sorted({
            re.sub(r"\s+", " ", ing.strip())
            for ing in re.split(r"[\n,;:+&|/]", text)
            if len(ing.strip()) > 2 and not ing.strip().isdigit()
        })

    @staticmethod
    def calculate_risk(harmful: List[str], substances: dict) -> str:
        """Risk seviyesi hesaplama"""
        if not harmful:
            return "Low"
        
        score = sum(
            Config.RISK_WEIGHTS.get(substances['harmful'].get(ing, {}).get("severity", "Low"), 0) +
            Config.RISK_WEIGHTS.get(substances['harmful'].get(ing, {}).get("category", "Other"), 0)
            for ing in harmful
        )
        return "Very High" if score >= 10 else "High" if score >=7 else "Medium" if score >=4 else "Low"

# ------------------------------
# Pipeline
# ------------------------------
class AnalysisPipeline:
    def __init__(self):
        self.substances = Config.load_substances()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ana işlem akışı"""
        # Ön işlemler
        df = self._preprocess(df)
        
        # Batch işleme
        results = []
        batches = self._create_batches(df)
        
        with ThreadPoolExecutor(max_workers=Config.get_max_workers()) as executor:
            futures = [executor.submit(self.process_batch, batch) for batch in batches]
            
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logging.error(f"Batch processing failed: {str(e)}")
                    results.append(pd.DataFrame())

        # Sonuçları birleştir
        return self._combine_results(df, pd.concat(results))

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Veriyi hazırlama"""
        df = df.dropna(subset=['barcode', 'name']).copy()
        df['ingredients'] = df['ingredients'].fillna('').apply(DataProcessor.parse_ingredients)
        return df

    def _create_batches(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Dinamik batch oluşturma"""
        batch_size = Config.get_batch_size()
        return [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(SafetyAnalyzer.analyze_ingredients, ingredients) 
                    for ingredients in batch['ingredients']]
            analyses = [f.result() for f in futures]
        return self._format_results(batch, analyses)

    def _format_results(self, batch: pd.DataFrame, analyses: List[dict]) -> pd.DataFrame:
        """Sonuçları formatlama"""
        result = batch.copy()
        # Her ürün için ayrı analiz
        result['harmful_ingredients'] = [a.get('harmful_ingredients', []) for a in analyses]
        result['safe_ingredients'] = [a.get('safe_ingredients', []) for a in analyses]
        result['description'] = [a.get('description', '') for a in analyses]
        
        return result

    def _combine_results(self, original: pd.DataFrame, processed: pd.DataFrame) -> pd.DataFrame:
        """Sonuçları birleştirme"""
        return original.merge(
            processed[['barcode', 'harmful_ingredients', 'safe_ingredients', 'description', 'risk_level']],
            on='barcode',
            how='left'
        )
    
# ------------------------------
# Model Kontrol Fonksiyonları (Yeni Eklenen)
# ------------------------------
def check_ollama_connection() -> bool:
    """Ollama servisinin çalıştığını doğrular"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Ollama connection failed: {str(e)}")
        return False

def verify_model() -> bool:
    """Doğru modelin yüklü olduğunu kontrol eder"""
    try:
        models = ollama.list().get('models', [])
        
        logging.info("Mevcut Modeller:")
        for model in models:
            # Model bilgilerini doğru anahtarla al
            model_name = model.get('model', 'UNKNOWN')
            logging.info(f"- {model_name}")

        # Model isimlerinde 'mistral' ara (büyük/küçük harf duyarsız)
        return any('mistral' in model.get('model', '').lower() for model in models)
    except Exception as e:
        logging.error(f"Model doğrulama hatası: {str(e)}")
        return False

# ------------------------------
# Execution
# ------------------------------
def main():
    try:
        # Test
        test_data = pd.DataFrame({
            'barcode': ['TEST123', 'TEST456'],
            'name': ['Test Product 1', 'Test Product 2'],
            'ingredients': [
                'water, sodium lauryl sulfate, parabens',
                'organic coconut oil, vitamin E'
            ]
        })
        
        pipeline = AnalysisPipeline()
        test_result = pipeline.process(test_data)
        logging.info("Test results:\n%s", test_result)
        
        # Ana işlem
        input_file = os.path.join(Config.INPUT_PATH, "processed_openfoodfacts.parquet")
        output_file = os.path.join(Config.OUTPUT_PATH, "pre_train_results.csv")
        
        df = pd.read_parquet(input_file)
        result = pipeline.process(df)
        
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
        result.to_csv(output_file, index=False)
        logging.info("Analysis completed. Results saved to: %s", output_file)
        
    except Exception as e:
        logging.critical("Pipeline execution failed: %s", str(e), exc_info=True)
        raise



# ------------------------------
# Main Execution (Güncellenmiş)
# ------------------------------
if __name__ == "__main__":
    try:
        # 1. Ollama bağlantı ve model kontrolleri
        logging.info("🔍 Sistem kontrolleri başlatılıyor...")
        
        if not check_ollama_connection():
            logging.critical("❌ Ollama servisi çalışmıyor! 'ollama serve' komutu ile başlatın.")
            exit(1)
            
        if not verify_model():
            logging.critical("❌ Mistral modeli yüklü değil! Komut: 'ollama pull mistral'")
            exit(1)

        # 2. Test analizi
        logging.info("🧪 Test analizi başlatılıyor...")
        test_ingredients = ["water", "sodium lauryl sulfate", "parabens"]
        test_result = SafetyAnalyzer.analyze_ingredients(test_ingredients)
        
        if "error" in test_result:
            logging.error("❌ Test başarısız! Örnek yanıt:")
            print(json.dumps(test_result, indent=2))
            exit(1)
            
        # 3. Ana işlem
        logging.info("🚀 Ana analiz başlatılıyor...")
        
        # Yol kontrolleri
        Config.validate_paths()
        input_path = os.path.abspath(os.path.join(Config.INPUT_PATH, "processed_openfoodfacts.parquet"))
        output_path = os.path.abspath(os.path.join(Config.OUTPUT_PATH, "pre_train_results.csv"))
        
        # Veriyi yükle ve işle
        df = pd.read_parquet(input_path)
        pipeline = AnalysisPipeline()
        result = pipeline.process(df)
        
        # Sonuçları kaydet
        result.to_csv(output_path, index=False)
        logging.info(f"✅ Başarıyla kaydedildi: {output_path}")
        print(f"\n📂 Çıktı Dosya Yolu:\n{output_path}")

    except Exception as e:
        logging.critical(f"⛔ Kritik hata: {str(e)}", exc_info=True)
        print("\n🆘 Çözüm adımları:")
        print("1. Ollama servisinin çalıştığından emin olun: 'ollama serve'")
        print("2. Dosya izinlerini kontrol edin: 'chmod 755 ./results'")
        print("3. Input dosyasının varlığını doğrulayın")
        exit(1)

