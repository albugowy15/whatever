from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 


rf_model_path = os.path.join(BASE_DIR, 'model', 'random_forest_1.0.pkl')
rf_model = joblib.load(rf_model_path)

app = FastAPI()

class InputData(BaseModel):
    heart_rate: int
    systolic_blood_pressure: int
    diastolic_blood_pressure: int
    body_temp: float

label_mapping = {
    "Relaxed": 0,
    "Calm": 1,
    "Tense": 2,
    "Stressed": 3
}

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict/rf/")
async def predict_stress_level(input_data: InputData):
    features = [input_data.heart_rate, input_data.systolic_blood_pressure, input_data.diastolic_blood_pressure, input_data.body_temp]
    prediction = rf_model.predict([features])
    print(prediction)
    return {"stress_level_index": "none", "stress_level_label": "none"}