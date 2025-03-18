import pandas as pd
import joblib
import ast
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("/Users/konak/Downloads/product-analysis/backend/ai-analysis/final3.csv")
    
    # Harmful_ingredients'i liste formatına çevir (Multi-label için)
    df["harmful_ingredients"] = df["harmful_ingredients"].apply(
        lambda x: ast.literal_eval(x) if x != "['None']" else []
    )
    
    # Tüm zararlı bileşenleri topla ve MultiLabelBinarizer ile kodla
    all_harmful = [item for sublist in df["harmful_ingredients"] for item in sublist]
    unique_harmful = list(set(all_harmful))
    
    mlb = MultiLabelBinarizer(classes=unique_harmful)
    harmful_encoded = mlb.fit_transform(df["harmful_ingredients"])
    
    # Diğer kategorik sütunlar için LabelEncoder
    toxicity_encoder = LabelEncoder()
    allergen_encoder = LabelEncoder()
    regulatory_encoder = LabelEncoder()
    
    df["toxicity_score"] = toxicity_encoder.fit_transform(df["toxicity_score"])
    df["allergen_risk"] = allergen_encoder.fit_transform(df["allergen_risk"])
    df["regulatory_status"] = regulatory_encoder.fit_transform(df["regulatory_status"])
    
    # Encoder'ları kaydet
    joblib.dump(mlb, "harmful_mlb.joblib")
    joblib.dump(toxicity_encoder, "toxicity_encoder.joblib")
    joblib.dump(allergen_encoder, "allergen_encoder.joblib")
    joblib.dump(regulatory_encoder, "regulatory_encoder.joblib")
    
    return df, harmful_encoded, (mlb, toxicity_encoder, allergen_encoder, regulatory_encoder)

def split_data(df, harmful_encoded):
    # Metin verisini birleştir
    X_text = df["ingredients"] + " " + df["general_health_assessment"]
    
    # Hedef değişkenleri ayır
    y_harmful = harmful_encoded
    y_toxicity = df["toxicity_score"].values
    y_allergen = df["allergen_risk"].values
    y_regulatory = df["regulatory_status"].values
    
    # Veriyi böl
    (
        X_train, 
        X_test, 
        y_harmful_train, 
        y_harmful_test,
        y_toxicity_train,
        y_toxicity_test,
        y_allergen_train,
        y_allergen_test,
        y_regulatory_train,
        y_regulatory_test
    ) = train_test_split(
        X_text,
        y_harmful,
        y_toxicity,
        y_allergen,
        y_regulatory,
        test_size=0.2,
        random_state=42
    )
    
    # Eğitim ve test verilerini dict olarak döndür
    y_train = {
        "harmful_output": y_harmful_train,
        "toxicity_output": y_toxicity_train,
        "allergen_output": y_allergen_train,
        "regulatory_output": y_regulatory_train
    }
    
    y_test = {
        "harmful_output": y_harmful_test,
        "toxicity_output": y_toxicity_test,
        "allergen_output": y_allergen_test,
        "regulatory_output": y_regulatory_test
    }
    
    return X_train, X_test, y_train, y_test
def prepare_bert_data(df):
    # Metinleri birleştir
    texts = df["ingredients"] + " [SEP] " + df["general_health_assessment"]
    
    # MultiLabelBinarizer ile zararlı bileşenleri kodla
    mlb = MultiLabelBinarizer()
    harmful_encoded = mlb.fit_transform(df["harmful_ingredients"])
    
    # Diğer etiketleri LabelEncoder ile kodla
    toxicity_encoder = LabelEncoder()
    allergen_encoder = LabelEncoder()
    regulatory_encoder = LabelEncoder()
    
    toxicity_labels = toxicity_encoder.fit_transform(df["toxicity_score"])
    allergen_labels = allergen_encoder.fit_transform(df["allergen_risk"])
    regulatory_labels = regulatory_encoder.fit_transform(df["regulatory_status"])
    
    # Encoder'ları kaydet
    joblib.dump(mlb, "bert_harmful_mlb.joblib")
    joblib.dump(toxicity_encoder, "bert_toxicity_encoder.joblib")
    joblib.dump(allergen_encoder, "bert_allergen_encoder.joblib")
    joblib.dump(regulatory_encoder, "bert_regulatory_encoder.joblib")
    
    return texts, (harmful_encoded, toxicity_labels, allergen_labels, regulatory_labels)
def prepare_bert_data(df):
    # Metinleri birleştir
    texts = df["ingredients"] + " [SEP] " + df["general_health_assessment"]
    
    # Zararlı bileşenleri listeye çevir
    df["harmful_ingredients"] = df["harmful_ingredients"].apply(
        lambda x: ast.literal_eval(x) if x != "['None']" else []
    )
    
    # MultiLabelBinarizer ile kodla
    mlb = MultiLabelBinarizer()
    harmful_encoded = mlb.fit_transform(df["harmful_ingredients"])
    
    # Diğer etiketleri kodla
    toxicity_encoder = LabelEncoder()
    allergen_encoder = LabelEncoder()
    regulatory_encoder = LabelEncoder()
    
    toxicity_labels = toxicity_encoder.fit_transform(df["toxicity_score"])
    allergen_labels = allergen_encoder.fit_transform(df["allergen_risk"])
    regulatory_labels = regulatory_encoder.fit_transform(df["regulatory_status"])
    
    # Encoder'ları kaydet
    joblib.dump(mlb, "bert_harmful_mlb.joblib")
    joblib.dump(toxicity_encoder, "bert_toxicity_encoder.joblib")
    joblib.dump(allergen_encoder, "bert_allergen_encoder.joblib")
    joblib.dump(regulatory_encoder, "bert_regulatory_encoder.joblib")
    
    return texts, (harmful_encoded, toxicity_labels, allergen_labels, regulatory_labels)