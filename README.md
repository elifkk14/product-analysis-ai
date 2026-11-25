# 📌 Product Content Analysis AI

🚀 **AI-powered product content analysis tool**  
🔍 Scans ingredients from **barcode/QR** or manual input and classifies them as **safe or harmful**  
🛠️ **Tech Stack:** Python, FastAPI, Next.js, Hugging Face, PostgreSQL  

---

## 🎯 Features

✅ **Ingredient Analysis** – Classifies product ingredients into **safe, harmful, or unknown**  
✅ **AI Model** – Uses **NLP & Machine Learning** to analyze product data  
✅ **Barcode/QR Support** – Scan product packaging for automatic data retrieval  
✅ **Database Support** – Stores past searches for future reference  

---

## 🖼️ Screenshots
<img width="1489" alt="Screenshot 2025-03-25 at 21 58 50" src="https://github.com/user-attachments/assets/b70af428-f60e-4969-beea-5b1d0f842d1e" />
<img width="1496" alt="Screenshot 2025-03-25 at 21 59 47" src="https://github.com/user-attachments/assets/79ad9322-094a-4300-a301-d3ae37586b72" />
<img width="1503" alt="Screenshot 2025-03-25 at 21 59 39" src="https://github.com/user-attachments/assets/1794fb6c-4f3e-4215-8033-8ffc33bd8803" />
<img width="1502" alt="Screenshot 2025-03-25 at 21 59 32" src="https://github.com/user-attachments/assets/f870022f-d37d-4b2f-9c10-89710abd2a51" />
<img width="1503" alt="Screenshot 2025-03-25 at 21 59 19" src="https://github.com/user-attachments/assets/9bd697f2-c972-4b7f-80d5-ca270016a72c" />
<img width="1501" alt="Screenshot 2025-03-25 at 21 59 10" src="https://github.com/user-attachments/assets/96b6a772-5497-4746-b04b-c0aaa46bc2d0" />
<img width="1495" alt="Screenshot 2025-03-25 at 21 59 01" src="https://github.com/user-attachments/assets/000c61af-17e8-47fa-b23f-e774d8086def" />

---

## 🧭 How It Works

```
OpenFoodFacts / OpenBeautyFacts dumps
        │  data_collection.py   (download with retry/backoff, chunked processing)
        ▼
processed_openfoodfacts.parquet  (barcode, name, brand, ingredients)
        │  optimized_pre_train.py / pre_train.py / last_pre_train.py
        │  (cleaning, balancing, LLM-assisted labeling via a local Mistral model)
        ▼
Labeled training data  ──►  BERT + LSTM models
        │
        ▼
FastAPI backend (/analyze)  ──►  Next.js frontend
```

### Data collection (`backend/ai-analysis/data_collection.py`)
- Downloads the full **OpenFoodFacts** (CSV) and **OpenBeautyFacts** (JSONL) product dumps in parallel.
- Uses a `requests` session with **automatic retries and exponential backoff** for `429` and `5xx` responses.
- Streams the large gzipped CSV and processes it in **100,000-row chunks** with pandas to keep memory usage low.
- Keeps only `barcode`, `name`, `brand` and `ingredients`, drops incomplete rows, and stores **barcodes as strings** so leading zeros are not lost.
- Saves the result as **Parquet** (or JSONL).

### Labeling pipeline
- Cleans the data (removes duplicates and very short ingredient texts) and builds a balanced sample of food and cosmetic products.
- Detects non-English ingredient lists and translates them.
- Sends ingredient lists to a local **Mistral** model served by **Ollama** and validates each JSON answer against a schema, with retries.
- Saves progress to a **checkpoint** file so long runs can resume where they stopped.

### API (`backend/main.py`)
- Combines a fine-tuned **BERT** model and an **LSTM** model (ensemble) to detect harmful ingredients and estimate a toxicity level.
- Exposes a single `POST /analyze` endpoint.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/elifkk14/product-analysis-ai.git
cd product-analysis-ai
```

### 2. Install backend dependencies (Python 3.11)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the trained models

Trained model files are **not included** in the repository because of their size. Place them under `backend/ai-analysis/deep_models/`:

```
backend/ai-analysis/deep_models/
├── bert_multi_output/        # fine-tuned BERT model + tokenizer
├── bert/bert_encoders.joblib # label encoders
└── ltsm/
    ├── lstm_model.h5
    └── lstm_tokenizer.joblib
```

To keep them somewhere else, set the `MODEL_DIR` environment variable:

```bash
export MODEL_DIR=/path/to/deep_models
```

### 4. Run the API

```bash
python main.py
```

The API runs on `http://localhost:8000`. Interactive docs: `http://localhost:8000/docs`.

---

## 🔌 API Usage

**Request**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ingredients": "water, sodium lauryl sulfate, parabens"}'
```

**Response**

```json
{
  "status": "success",
  "analysis": {
    "harmful_ingredients": ["..."],
    "safe_ingredients": ["..."],
    "toxicity_level": "...",
    "models_used": ["BERT", "LSTM"]
  }
}
```

Ingredients are expected as a comma-separated list (`"ing1, ing2, ing3"`).

---

## 🔄 Rebuilding the Dataset (optional)

```bash
cd backend/ai-analysis

# 1. Download and process product data → data/processed_openfoodfacts.parquet
python data_collection.py

# 2. Label the data with a local LLM (requires Ollama)
ollama serve
ollama pull mistral:7b-instruct-q4_K_M
python optimized_pre_train.py
```

> The full OpenFoodFacts dump is several gigabytes. Make sure you have enough disk space before running `data_collection.py`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
