import pandas as pd
import csv
import re

# Girdi ve çıktı dosya yolları
input_file = '/Users/konak/Downloads/product-analysis/backend/ai-analysis/final2.csv'
output_file = '/Users/konak/Downloads/product-analysis/backend/ai-analysis/final3.csv'

# CSV dosyasını oku
df = pd.read_csv(input_file)

# Tüm sütunlar açısından tamamen aynı olan duplicate satırları kaldır
df = df.drop_duplicates()

# Veride bulunan fazla çift tırnakları temizleyip, metni tam olarak bir çift tırnakla sarmalayan fonksiyon
def standardize_text(val):
    if isinstance(val, str):
        # Başta ve sonda bulunan bir veya daha fazla çift tırnak karakterini kaldırır
        cleaned = re.sub(r'^"+', '', val)
        cleaned = re.sub(r'"+$', '', cleaned)
        # Temizlenmiş değeri tek çift tırnakla sarmalar
        return f'"{cleaned}"'
    return val

# Standart hale getirilecek metin sütunları (toxicity_score, allergen_risk, regulatory_status hariç)
textual_columns = ['ingredients', 'safe_ingredients', 'harmful_ingredients', 'general_health_assessment']

# İlgili sütunlardaki verileri temizle
for col in textual_columns:
    if col in df.columns:
        df[col] = df[col].apply(standardize_text)

# Yazdırırken, metin sütunlarımızdaki manuel eklediğimiz çift tırnakların korunması için quoting'i kapatıyoruz
df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

print("CSV dosyası istenilen formatta oluşturuldu:", output_file)
