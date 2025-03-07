import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

MAX_FEATURES = 200000
SEQUENCE_LENGTH = 1800


def load_data():
    df = pd.read_csv(os.path.join("..", "data", "train.csv"))
    X = df["comment_text"]
    y = df[df.columns[2:]].values
    return X, y


vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES, output_sequence_length=SEQUENCE_LENGTH, output_mode="int"
)
X, y = load_data()
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache().shuffle(160000).batch(16).prefetch(8)

train = dataset.take(int(len(dataset) * 0.7))
val = dataset.skip(int(len(dataset) * 0.7)).take(int(len(dataset) * 0.2))
test = dataset.skip(int(len(dataset) * 0.9)).take(int(len(dataset) * 0.1))


def get_datasets():
    return train, val, test


def get_vectorizer():
    return vectorizer
