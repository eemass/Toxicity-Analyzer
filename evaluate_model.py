import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from model import load_or_train_model
from data_loader import get_datasets


def evaluate_model():
    model = load_or_train_model()
    _, _, test = get_datasets()
    pre, re, acc = Precision(), Recall(), CategoricalAccuracy()

    for batch in test.as_numpy_iterator():
        X_true, y_true = batch
        yhat = model.predict(X_true)
        y_true, yhat = y_true.flatten(), yhat.flatten()
        pre.update_state(y_true, yhat)
        re.update_state(y_true, yhat)
        acc.update_state(y_true, yhat)

    results = {
        "Precision": pre.result().numpy(),
        "Recall": re.result().numpy(),
        "Accuracy": acc.result().numpy(),
    }

    print(results)
    return results


if __name__ == "__main__":
    evaluate_model()
