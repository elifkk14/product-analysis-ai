import json
import sys
import os
import pandas as pd
import requests
import logging
import concurrent.futures
from langdetect import detect, DetectorFactory, LangDetectException
from tqdm import tqdm
import time
import re

# M1 Optimizasyon Ayarları
DetectorFactory.seed = 0
MAX_WORKERS = 4
BATCH_SIZE = 50
CHECKPOINT_INTERVAL = 100

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

class Config:
    INPUT_CSV = "/Users/konak/Downloads/product-analysis/backend/ai-analysis/optimized_training_data_fixed.csv"
    OUTPUT_CSV = "last_output.csv"
    API_ENDPOINT = "http://localhost:11434/api/generate"
    MODEL_NAME = "mistral:7b-instruct-q2_K"
    TIMEOUT = 180
    MAX_RETRIES = 3
    MODEL_PARAMS = {
        "temperature": 0.3,
        "num_ctx": 1024,
        "repeat_penalty": 1.1,
        "max_tokens": 100
    }

# Logging Yapılandırması
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def print_banner():
    print(f"{Colors.CYAN}\nAI Data Processor v3.4 (M1 Optimized)\n{Colors.RESET}")

def is_valid_for_translation(text: str) -> bool:
    return bool(text and text.strip() and re.search(r'[a-zA-Z]', text))

def safe_get(row, key, default=""):
    try:
        return str(row[key]) if pd.notna(row[key]) else default
    except (KeyError, TypeError):
        return default

def query_model(prompt: str) -> str:
    for retry in range(Config.MAX_RETRIES):
        try:
            response = requests.post(
                Config.API_ENDPOINT,
                json={"prompt": prompt, "model": Config.MODEL_NAME, **Config.MODEL_PARAMS},
                timeout=Config.TIMEOUT,
                stream=True
            )
            if response.status_code == 200:
                full_response = []
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            json_chunk = json.loads(line)
                            full_response.append(json_chunk.get("response", ""))
                            if json_chunk.get("done", False):
                                break
                        except json.JSONDecodeError as e:
                            logging.error(f"{Colors.RED}JSON error: {e}{Colors.RESET}")
                result = ''.join(full_response).strip()
                return result if result else "Unknown"
        except requests.exceptions.RequestException as e:
            logging.error(f"{Colors.RED}Connection error (Retry {retry+1}): {e}{Colors.RESET}")
            time.sleep(5 * (retry + 1))
    return "Unknown"

def translate_if_needed(text: str) -> str:
    if not is_valid_for_translation(text):
        return text
    try:
        DetectorFactory.seed = 0
        lang = detect(text)
        if lang != "en":
            translated = query_model(f"Translate to English: {text[:500]}")
            logging.info(f"{Colors.CYAN}Translated: '{text[:30]}' → '{translated[:30]}'{Colors.RESET}")
            return translated
    except LangDetectException as e:
        logging.error(f"{Colors.RED}Language error: {e}{Colors.RESET}")
    return text

def generate_ai_content(context: str, field_type: str) -> str:
    prompts = {
        'safe_ingredients': "List 3-5 safe ingredients for {context}. Respond only with comma-separated values.",
        'harmful_ingredients': "List potential harmful ingredients for {context}. Respond only with comma-separated values.",
        'allergen_list': "Predict allergens for {context}. Respond only with comma-separated values.",
        'regulatory_status': "Regulatory status for {context} in one short sentence."
    }
    prompt = prompts.get(field_type, "").format(context=context)
    response = query_model(prompt).strip('"')
    return response if response else "Unknown"

def fix_list_format(text: str) -> str:
    if not text or text.lower() in ["unknown", "nan", "null", "[]"]:
        return "[]"
    if not (text.startswith("[") and text.endswith("]")):
        parts = [x.strip() for x in re.split(r'[,;]', text) if x.strip()]
        return str(parts[:5])
    return text

def process_row(row: pd.Series) -> pd.Series:
    new_row = row.copy()
    try:
        product = safe_get(new_row, 'product_name', 'Unknown Product')
        category = safe_get(new_row, 'category', 'General')
        context = f"{product} ({category})"
        
        # AI Content Generation
        fields = ['safe_ingredients', 'harmful_ingredients', 'allergen_list', 'regulatory_status']
        for field in fields:
            current_value = safe_get(new_row, field)
            new_value = generate_ai_content(context, field)
            new_row[field] = new_value
            logging.info(f"{Colors.GREEN}Generated {field}: {new_value[:50]}...{Colors.RESET}")
        
        # Translation
        for col in new_row.index:
            original = str(new_row[col])
            translated = translate_if_needed(original)
            new_row[col] = translated
        
        # Fix Lists
        for field in ['safe_ingredients', 'harmful_ingredients', 'allergen_list']:
            fixed = fix_list_format(safe_get(new_row, field))
            new_row[field] = fixed
        
        return new_row
    except Exception as e:
        logging.error(f"{Colors.RED}Row error: {e}{Colors.RESET}")
        return new_row

def main():
    print_banner()
    start_time = time.time()
    
    try:
        df = pd.read_csv(Config.INPUT_CSV, dtype={'product_name': 'string', 'category': 'string'}).fillna('Unknown')
        checkpoint_file = "/Users/konak/Downloads/product-analysis/backend/ai-analysis/checkpoint.csv"

        # Debug: Giriş dosyasını kontrol et
        logging.info(f"{Colors.CYAN}Input CSV loaded with {len(df)} rows.{Colors.RESET}")
        
        # Checkpoint dosyasını kontrol et
        if not os.path.exists(checkpoint_file):
            logging.info(f"{Colors.GREEN}Checkpoint file not found. Creating new checkpoint file.{Colors.RESET}")
            pd.DataFrame(columns=df.columns).to_csv(checkpoint_file, index=False)
            processed_count = 0
            results = []
        else:
            checkpoint_df = pd.read_csv(checkpoint_file)
            processed_count = len(checkpoint_df)
            results = checkpoint_df.to_dict('records')
            logging.info(f"{Colors.CYAN}Resuming from checkpoint: {processed_count} rows already processed.{Colors.RESET}")
        
        remaining_df = df.iloc[processed_count:]
        total_rows = len(remaining_df)
        logging.info(f"{Colors.CYAN}Processing {total_rows} remaining rows...{Colors.RESET}")

        if total_rows == 0:
            logging.info(f"{Colors.GREEN}No rows to process. Exiting.{Colors.RESET}")
            return
        
        # Processing
        with tqdm(total=len(remaining_df), desc=f"{Colors.GREEN}Processing") as pbar:
            batches = [remaining_df.iloc[i:i+BATCH_SIZE] for i in range(0, len(remaining_df), BATCH_SIZE)]
            for batch in batches:
                batch_results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {executor.submit(process_row, row): idx for idx, row in batch.iterrows()}
                    for future in concurrent.futures.as_completed(futures):
                        batch_results.append(future.result())
                        pbar.update(1)
                results.extend(batch_results)
                
                # Checkpoint
                if len(results) % CHECKPOINT_INTERVAL == 0:
                    pd.DataFrame(results).to_csv(checkpoint_file, index=False)
                    logging.info(f"{Colors.CYAN}Checkpoint saved at {len(results)} rows.{Colors.RESET}")
        
        # Final Save
        pd.DataFrame(results).to_csv(Config.OUTPUT_CSV, index=False)
        logging.info(f"{Colors.GREEN}Process completed in {(time.time()-start_time)/60:.2f} minutes.{Colors.RESET}")
    
    except Exception as e:
        logging.error(f"{Colors.RED}Fatal error: {e}{Colors.RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()