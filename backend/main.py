from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np
import uvicorn
from tensorflow.keras.models import load_model

app = FastAPI()

# ========================= MODEL YÜKLEME =========================
# BERT Modeli
BERT_MODEL = load_model(
    "/Users/konak/Downloads/product-analysis/backend/ai-analysis/deep_models/bert_multi_output",
    custom_objects={"TFBertModel": TFBertModel},  # Hugging Face BERT katmanı için
    compile=False
)

# LSTM Modeli
LSTM_MODEL = load_model(
    "/Users/konak/Downloads/product-analysis/backend/ai-analysis/deep_models/ltsm/lstm_model.h5"
)

# Tokenizer ve Encoder'lar
BERT_TOKENIZER = BertTokenizer.from_pretrained(
    "/Users/konak/Downloads/product-analysis/backend/ai-analysis/deep_models/bert_multi_output"  # Yerel tokenizer
)
LSTM_TOKENIZER = joblib.load(
    "/Users/konak/Downloads/product-analysis/backend/ai-analysis/deep_models/ltsm/lstm_tokenizer.joblib"
)
ENCODERS = joblib.load(
    "/Users/konak/Downloads/product-analysis/backend/ai-analysis/deep_models/bert/bert_encoders.joblib"
)

# ========================= CORS AYARLARI =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================= REQUEST MODELİ =========================
class AnalysisRequest(BaseModel):
    ingredients: str

# ========================= TAHMİN FONKSİYONU =========================
def ensemble_predict(ingredients: str):
    # BERT Tahmini (MAX_LENGTH=128)
    bert_inputs = BERT_TOKENIZER(
        ingredients,
        max_length=128,  # Eğitimle aynı değer
        truncation=True,
        padding="max_length",
        return_tensors="tf"
    )
    
    # Model Çıktıları (input_ids ve attention_mask KULLAN)
    bert_outputs = BERT_MODEL.predict({
        "input_ids": bert_inputs["input_ids"],
        "attention_mask": bert_inputs["attention_mask"]
    })
    
    # Çıktıları Ayrıştır
    harmful_output = bert_outputs[0]  # (1, N)
    toxicity_output = bert_outputs[1]  # (1, toxicity_classes)

    # LSTM Tahmini (MAXLEN=128)
    lstm_seq = LSTM_TOKENIZER.texts_to_sequences([ingredients])
    lstm_padded = pad_sequences(lstm_seq, maxlen=128, padding="post")  # Eğitimle aynı
    lstm_outputs = LSTM_MODEL.predict(lstm_padded)
    
    # Sonuçları Birleştir
    harmful_bert = ENCODERS["harmful_mlb"].inverse_transform(harmful_output > 0.5)[0]
    harmful_lstm = ENCODERS["harmful_mlb"].inverse_transform(lstm_outputs[0] > 0.5)[0]
    
    final_harmful = list(set(harmful_bert) | set(harmful_lstm))
    
    # Toxicity için Ortalama
    final_toxicity = ENCODERS["toxicity_encoder"].inverse_transform([
        np.argmax((toxicity_output[0] + lstm_outputs[1][0])/2)
    ])[0]

    return {
        "harmful": final_harmful,
        "toxicity": final_toxicity,
        "used_models": ["BERT", "LSTM"]
    }

# ========================= API ENDPOINT =========================
@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    try:
        ingredients = request.ingredients.lower().strip()
        analysis = ensemble_predict(ingredients)
        return {
            "status": "success",
            "analysis": {
                "harmful_ingredients": analysis["harmful"],
                "safe_ingredients": [ing for ing in ingredients.split(", ") if ing not in analysis["harmful"]],
                "toxicity_level": analysis["toxicity"],
                "models_used": analysis["used_models"]
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "details": "Input format: 'ing1, ing2, ing3'"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)