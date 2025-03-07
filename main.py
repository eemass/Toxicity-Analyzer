import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import load_or_train_model
from data_loader import get_vectorizer

app = FastAPI()
model = load_or_train_model()
vectorizer = get_vectorizer()


class InputText(BaseModel):
    text: str


@app.post("/predict")
def predict_toxicity(input_text: InputText):
    try:
        categories = [
            "Toxic",
            "Severe Toxic",
            "Obscene",
            "Threat",
            "Insult",
            "Identity Hate",
        ]

        input_vector = vectorizer([input_text.text])
        input_vector = np.array(input_vector)
        input_vector = np.reshape(input_vector, (1, 1800))

        raw_predictions = model.predict(input_vector)[0]

        predictions = {
            categories[i]: bool(raw_predictions[i] >= 0.5)
            for i in range(len(categories))
        }

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
