import pandas as pd

file_path = "/Users/konak/Downloads/product-analysis/backend/ai-analysis/data/cleaned_dataset_final.csv"
df = pd.read_csv(file_path)

# 📊 Veri setinin genel bilgisi
print("🔍 Veri Seti Genel Bilgi:")
print(df.info())

# 📊 Kategori dağılımı
print("\n🔍 Zararlı İçerik Dağılımı:")
print(df["harmful"].value_counts())

# 📊 İlk 5 Satır
print("\n🔍 Örnek Veri:")
print(df.head())

# İçerik örneklerini kontrol et
print("\n🔍 İçerik Örnekleri:")
print(df["ingredients"].sample(10))

# Boş içerik kontrolü
missing_ingredients = df["ingredients"].isna().sum()
print(f"\n⚠️ Eksik İçerik Sayısı: {missing_ingredients}")

# Kategori bazlı dağılımı incele
if "category" in df.columns:
    print("\n🔍 Kategori Dağılımı:")
    print(df["category"].value_counts())
else:
    print("⚠️ Kategori sütunu bulunamadı.")
