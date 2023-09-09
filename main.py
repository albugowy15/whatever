from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import os
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 


rf_model_path = os.path.join(BASE_DIR, 'model', 'random_forest_1.0.pkl')
rf_model = joblib.load(rf_model_path)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    heart_rate: int
    systolic_blood_pressure: int
    diastolic_blood_pressure: int
    body_temp_c: float

labels = ["Relaxed", "Calm", "Tense", "Stressed"];

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict/rf")
async def predict_stress_level(input_data: InputData):
    incoming_data = pd.DataFrame({
        'heart_rate': [input_data.heart_rate],
        'systolic_blood_pressure': [input_data.systolic_blood_pressure],
        'diastolic_blood_pressure': [input_data.diastolic_blood_pressure],
        'body_temp_c': [input_data.body_temp_c]
    })
    prediction = rf_model.predict(incoming_data)
    print(prediction[0])
    print(type(prediction[0]))
    return {"stress_level_index": int(prediction[0]), "stress_level_label": labels[int(prediction[0])]}