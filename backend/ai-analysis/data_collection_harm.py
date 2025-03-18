import requests
import pandas as pd
import os

def fetch_fda_enforcement_data(limit=100):
    """
    openFDA API'sinden zararlı maddeleri içeren gıda geri çağırma raporlarını çeker.
    """
    food_url = f"https://api.fda.gov/food/enforcement.json?search=report_date:[20040101+TO+20231231]&limit={limit}"
    try:
        food_response = requests.get(food_url, timeout=10)
        food_response.raise_for_status()
        food_data = food_response.json().get("results", [])
        return food_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FDA enforcement data: {e}")
        return []

def fetch_cosmetic_data():
    """
    Kozmetik ürünlerde zararlı maddeleri içeren CSV dosyasını işler.
    """
    cosmetic_file_path = "/Users/konak/Downloads/product-analysis/backend/ai-analysis/data/cscpopendata_clean.csv"  # Güncellenmiş yol
    if not os.path.exists(cosmetic_file_path):
        print("Error: Cosmetic data file not found. Please check the file path.")
        return []
    try:
        df_cosmetic = pd.read_csv(cosmetic_file_path)
        harmful_cosmetics = df_cosmetic[["ProductName_Clean", "CompanyName_Clean", "BrandName_Clean", "PrimaryCategory_Clean", "SubCategory_Clean", "ChemicalName_Clean"]]
        harmful_cosmetics = harmful_cosmetics.rename(columns={
            "ProductName_Clean": "product_description",
            "CompanyName_Clean": "company_name",
            "BrandName_Clean": "brand_name",
            "PrimaryCategory_Clean": "primary_category",
            "SubCategory_Clean": "sub_category",
            "ChemicalName_Clean": "reason_for_recall"
        })
        harmful_cosmetics["category"] = "Cosmetic"
        return harmful_cosmetics.to_dict(orient="records")
    except Exception as e:
        print(f"Error processing cosmetic data: {e}")
        return []

def fetch_indirect_additives():
    """
    FDA'nın dolaylı katkı maddeleri listesini temizleyerek işler.
    """
    indirect_additives_file_path = "/Users/konak/Downloads/product-analysis/backend/ai-analysis/data/cleaned_chemichals.csv"
    if not os.path.exists(indirect_additives_file_path):
        print("Error: Indirect additives file not found. Please check the file path.")
        return []
    try:
        df_indirect = pd.read_csv(indirect_additives_file_path, encoding="ISO-8859-1")
        df_indirect = df_indirect.iloc[:, :2]  # Sadece CAS numarası ve kimyasal adı alınır
        df_indirect.columns = ["cas_number", "chemical_name"]
        df_indirect = df_indirect.dropna()
        df_indirect["category"] = "Indirect Additive"
        return df_indirect.to_dict(orient="records")
    except Exception as e:
        print(f"Error processing indirect additives data: {e}")
        return []

def process_fda_data(data):
    """
    Zararlı maddeleri içeren FDA ve kozmetik geri çağırma verilerini temizler ve işler.
    """
    harmful_data = []
    for item in data:
        harmful_data.append({
            "product_description": item.get("product_description", "N/A"),
            "reason_for_recall": item.get("reason_for_recall", "N/A"),
            "classification": item.get("classification", "N/A"),
            "recalling_firm": item.get("recalling_firm", "N/A"),
            "status": item.get("status", "N/A"),
            "country": item.get("country", "N/A"),
            "city": item.get("city", "N/A"),
            "state": item.get("state", "N/A"),
            "report_date": item.get("report_date", "N/A"),
            "category": item.get("category", "Food")
        })
    return harmful_data

def save_to_parquet(data, filename="/Users/konak/Downloads/product-analysis/backend/ai-analysis/data/harmful_substances.parquet"):
    """
    İşlenmiş zararlı maddeler verisini Parquet formatında kaydeder ve içeriğini kontrol eder.
    """
    if not data:
        print("No data available to save.")
        return
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_parquet(filename, index=False)
    print(f"Data saved to {filename}")
    
    # Parquet içeriğini kontrol etme
    print("Checking saved Parquet file...")
    df_check = pd.read_parquet(filename)
    print(df_check.info())
    print(df_check.head())

if __name__ == "__main__":
    print("Fetching FDA Enforcement Data...")
    fda_data = fetch_fda_enforcement_data(limit=500)
    cosmetic_data = fetch_cosmetic_data()
    indirect_additives_data = fetch_indirect_additives()
    combined_data = fda_data + cosmetic_data + indirect_additives_data
    processed_data = process_fda_data(combined_data)
    save_to_parquet(processed_data)
