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

# Run with Docker Compose:
docker compose build
docker compose up

Visit Swagger UI at http://localhost:8000/docs

Deploy to Render
1. Push your repo to GitHub

2. Create a new Web Service on Render

3. Select Docker as the environment

4. Connect your GitHub repository

5. Deploy and test your live API URL /docs

API Endpoints
* GET /health - Health check

* GET /info - Model info & feature list

* POST /predict - Predict heart disease (send JSON body with features)