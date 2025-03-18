import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np
from data_loader import load_data, split_data

# Veriyi yükle
df, harmful_encoded, encoders = load_data()
X_train, X_test, y_train, y_test = split_data(df, harmful_encoded)

# Hyperparameters
MAX_WORDS = 10000
MAX_LEN = 128  # ✅ 256 yerine 128 kullanıldı
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 10

# Tokenizer'ı oluştur ve metinleri vektörleştir
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

joblib.dump(tokenizer, "lstm_tokenizer.joblib")

# Metinleri sayısal diziye çevir ve 128 token ile pad'le
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post")  # ✅
X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post")    # ✅

# Model Architecture
input_layer = Input(shape=(MAX_LEN,))
embedding = Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM)(input_layer)
lstm1 = Bidirectional(LSTM(64, return_sequences=True))(embedding)
lstm2 = Bidirectional(LSTM(32))(lstm1)
dense = Dense(64, activation="relu")(lstm2)

# Çıktı Katmanları
output_harmful = Dense(
    harmful_encoded.shape[1],  # Zararlı bileşen sayısı
    activation="sigmoid", 
    name="harmful_output"
)(dense)

output_toxicity = Dense(
    len(encoders[1].classes_), 
    activation="softmax", 
    name="toxicity_output"
)(dense)

output_allergen = Dense(
    len(encoders[2].classes_), 
    activation="softmax", 
    name="allergen_output"
)(dense)

output_regulatory = Dense(
    len(encoders[3].classes_), 
    activation="softmax", 
    name="regulatory_output"
)(dense)

model = Model(
    inputs=input_layer,
    outputs=[output_harmful, output_toxicity, output_allergen, output_regulatory]
)

# Loss Fonksiyonları
model.compile(
    optimizer="adam",
    loss={
        "harmful_output": "binary_crossentropy",  # Multi-label
        "toxicity_output": "sparse_categorical_crossentropy",
        "allergen_output": "sparse_categorical_crossentropy",
        "regulatory_output": "sparse_categorical_crossentropy"
    },
    metrics=["accuracy"]
)

# Modeli Eğit
history = model.fit(
    X_train_padded,
    {
        "harmful_output": y_train["harmful_output"],
        "toxicity_output": y_train["toxicity_output"],
        "allergen_output": y_train["allergen_output"],
        "regulatory_output": y_train["regulatory_output"]
    },
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(
        X_test_padded,
        {
            "harmful_output": y_test["harmful_output"],
            "toxicity_output": y_test["toxicity_output"],
            "allergen_output": y_test["allergen_output"],
            "regulatory_output": y_test["regulatory_output"]
        }
    ),
    verbose=1
)

# Modeli Kaydet
model.save("lstm_model.h5")
print("Model basariyla kaydedildi!")