import csv

input_path = '/Users/konak/Downloads/product-analysis/backend/ai-analysis/final2.csv'
output_path = '/Users/konak/Downloads/product-analysis/backend/ai-analysis/final2_standardized.csv'

with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
     open(output_path, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
    
    # Header'ı oku ve yaz
    try:
        header = next(reader)
        writer.writerow(header)
    except StopIteration:
        raise ValueError("CSV file is empty")
    
    # Satırları işle
    for i, row in enumerate(reader):
        if len(row) != 7:
            print(f"⚠️ Satır {i+2}: Eksik/hatalı veri - {row}")
            continue
            
        try:
            # Veri temizliği için küçük düzeltmeler
            row[-1] = row[-1].replace('Pendig', 'Pending')  # Yaygın yazım hatası
            writer.writerow(row)
        except Exception as e:
            print(f"🚨 Satır {i+2} işlenemedi: {e}")

print("\n✅ Standardizasyon tamamlandı! Tutarsız satırlar konsolda gösterildi.")