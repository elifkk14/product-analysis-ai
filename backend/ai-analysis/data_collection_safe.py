import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_fda_scogs():
    """Loads FDA SCOGS CSV dataset and processes it correctly."""
    scogs_path = "data/SCOGS.csv"  # Ensure this path is correct

    try:
        # ✅ CSV Dosyasını yükle, doğru delimiter kullan
        df_scogs = pd.read_csv(scogs_path, sep=",", encoding="utf-8", skipinitialspace=True)

        # ✅ Sütun isimlerini dönüştür
        rename_mapping = {
            "GRAS Substance": "substance",
            "Other Names": "alternative_names",
            "SCOGS Report Number": "report_number",
            "CAS Reg. No. or other ID CODE": "cas_number",
            "Year of Report": "year",
            "SCOGS Type of Conclusion": "safety_level",
            "NTIS Accession Number": "ntis_accession"
        }

        df_scogs = df_scogs.rename(columns=rename_mapping)

        # ✅ Gerekli sütunları seç
        df_scogs = df_scogs[["substance", "safety_level"]]

        # ✅ Kategori sütunu eksik olduğu için varsayılan olarak ekleyelim
        df_scogs["category"] = "Food Additive"

        # ✅ Kaynak belirtelim
        df_scogs["source"] = "FDA-SCOGS"

        # ✅ Boş satırları kaldır
        df_scogs = df_scogs.dropna(subset=["substance"])

        return df_scogs[["substance", "category", "safety_level", "source"]]

    except Exception as e:
        print(f"Error loading SCOGS data: {e}")
        return pd.DataFrame()



def get_fda_gras():
    url = "https://www.fda.gov/food/gras-substances-scogs-database"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Parse table data
    gras_data = []
    table = soup.find('table', {'class': 'table-responsive'})
    if table:
        for row in table.find_all('tr')[1:]:  # Skip header row
            cols = row.find_all('td')
            if len(cols) > 0:
                gras_data.append({
                    "substance": cols[0].text.strip(),
                    "category": "Food Additive",
                    "safety_level": "GRAS",
                    "source": "FDA"
                })
    
    return pd.DataFrame(gras_data)

def get_openfoodfacts_safe():
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "action": "process",
        "tagtype_0": "categories",
        "tag_contains_0": "contains",
        "tag_0": "safe-ingredients",
        "json": 1,
        "page_size": 1000
    }
    
    response = requests.get(url, params=params).json()
    safe_ingredients = set()
    
    for product in response.get('products', []):
        if 'ingredients_analysis' in product:
            if 'en:non-vegan' not in product['ingredients_analysis_tags']:
                safe_ingredients.update(product.get('ingredients_tags', []))
    
    return pd.DataFrame({"substance": list(safe_ingredients), "category": "Safe Ingredient", "safety_level": "Safe", "source": "OpenFoodFacts"})

def save_safe_substances():
    df_scogs = get_fda_scogs()
    df_fda = get_fda_gras()
    df_openfoodfacts = get_openfoodfacts_safe()
    
    df_safe = pd.concat([df_scogs, df_fda, df_openfoodfacts], ignore_index=True)
    df_safe.to_parquet("data/safe_substances.parquet", index=False)
    print("Safe substances dataset saved successfully!")

if __name__ == "__main__":
    save_safe_substances()
