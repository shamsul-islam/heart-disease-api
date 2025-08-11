from fastapi import FastAPI
import joblib
import numpy as np
from app.schemas import HeartData

app = FastAPI()

# Load model
model = joblib.load("model/heart_model.joblib")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model_type": "RandomForestClassifier",
        "features": [
            "age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal"
        ]
    }

@app.post("/predict")
def predict(data: HeartData):
    features = np.array([
        data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
        data.restecg, data.thalach, data.exang, data.oldpeak, data.slope,
        data.ca, data.thal
    ]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {"heart_disease": bool(prediction)}
