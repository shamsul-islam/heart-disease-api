# ❤️ Heart Disease Prediction API — FastAPI + Docker + Render Deployment

A simple **FastAPI** application that serves predictions from a **Random Forest** classifier trained on the Heart Disease dataset.  
The focus of this project is **Dockerization** and **deployment**, not maximum model accuracy.

---

## 📂 Project Structure

heart_app/
│
├── app/
│ ├── main.py # FastAPI application
│ ├── schemas.py # Pydantic model for request validation
│
├── model/
│ └── heart_model.joblib # Pre-trained model
│
├── train_model.py # Script to train and save the model
├── requirements.txt # Python dependencies
├── Dockerfile # Docker build instructions
├── docker-compose.yml # Local Docker Compose setup
└── README.md # Project documentation


---

## 📊 Dataset
The dataset is from Kaggle:  
[Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
Binary classification: **1 = presence of heart disease**, **0 = absence**.

**Features used:**
age, sex, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal


---

## 🧠 Model Training
Run locally to train and save the model:
```bash
pip install -r requirements.txt
python train_model.py



