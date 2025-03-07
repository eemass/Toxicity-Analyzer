import os
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential
from data_loader import get_datasets

MAX_FEATURES = 200000
MODEL_PATH = os.path.join("model", "toxicity_analysis_model.keras")


def build_model():
    model = Sequential(
        [
            Embedding(MAX_FEATURES + 1, 32),
            Bidirectional(LSTM(32, activation="tanh")),
            Dense(128, activation="relu"),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(6, activation="sigmoid"),
        ]
    )
    model.compile(loss="BinaryCrossentropy", optimizer="Adam")
    return model


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)

    print("Model not found. Training a new model...")
    train, val, _ = get_datasets()
    model = build_model()
    model.fit(train, epochs=5, validation_data=val)
    model.save(MODEL_PATH)
    return model
