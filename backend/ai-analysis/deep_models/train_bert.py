import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from data_loader import prepare_bert_data
import pandas as pd
import numpy as np
import joblib

# Veriyi yükle ve hazırla
df = pd.read_csv("/Users/konak/Downloads/product-analysis/backend/ai-analysis/final3.csv")
texts, labels = prepare_bert_data(df)  # texts ve 4 etiket döner

# Etiketleri ayrıştır (DÜZELTME: labels tuple'ını aç)
harmful_labels, toxicity_labels, allergen_labels, regulatory_labels = labels

# BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize işlemi (TensorFlow tensörleri olarak)
encodings = tokenizer(
    texts.tolist(),
    max_length=128,
    padding="max_length",
    truncation=True,
    return_tensors="tf"
)

# Dataset oluşturma
dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"]
    },
    {
        "harmful_output": tf.convert_to_tensor(harmful_labels, dtype=tf.float32),
        "toxicity_output": tf.convert_to_tensor(toxicity_labels, dtype=tf.int32),
        "allergen_output": tf.convert_to_tensor(allergen_labels, dtype=tf.int32),
        "regulatory_output": tf.convert_to_tensor(regulatory_labels, dtype=tf.int32)
    }
)).batch(16)

# Model Mimarisi
input_ids = Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(128,), dtype=tf.int32, name="attention_mask")

bert_model = TFBertModel.from_pretrained("bert-base-uncased")
bert_output = bert_model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [CLS] token

# Çıktı Katmanları
harmful_output = Dense(
    harmful_labels.shape[1], 
    activation="sigmoid", 
    name="harmful_output"
)(bert_output)

toxicity_output = Dense(
    len(np.unique(toxicity_labels)), 
    activation="softmax", 
    name="toxicity_output"
)(bert_output)

allergen_output = Dense(
    len(np.unique(allergen_labels)), 
    activation="softmax", 
    name="allergen_output"
)(bert_output)

regulatory_output = Dense(
    len(np.unique(regulatory_labels)), 
    activation="softmax", 
    name="regulatory_output"
)(bert_output)

model = Model(
    inputs=[input_ids, attention_mask],
    outputs=[harmful_output, toxicity_output, allergen_output, regulatory_output]
)

# Loss Fonksiyonları
losses = {
    "harmful_output": BinaryCrossentropy(),
    "toxicity_output": SparseCategoricalCrossentropy(),
    "allergen_output": SparseCategoricalCrossentropy(),
    "regulatory_output": SparseCategoricalCrossentropy()
}

# Optimizer
optimizer = Adam(learning_rate=2e-5)

# Modeli Derle
model.compile(
    optimizer=optimizer,
    loss=losses,
    metrics=["accuracy"]
)

# Modeli Eğit
history = model.fit(
    dataset,
    epochs=3,
    verbose=1
)

# Modeli Kaydet
model.save("deep_models/bert_multi_output")

tokenizer.save_pretrained("deep_models/bert_multi_output")

# Encoder'ları kaydet
joblib.dump({
    "harmful_mlb": joblib.load("bert_harmful_mlb.joblib"),
    "toxicity_encoder": joblib.load("bert_toxicity_encoder.joblib"),
    "allergen_encoder": joblib.load("bert_allergen_encoder.joblib"),
    "regulatory_encoder": joblib.load("bert_regulatory_encoder.joblib")
}, "deep_models/bert_encoders.joblib")

print("Eğitim başarıyla tamamlandı! 🎉")