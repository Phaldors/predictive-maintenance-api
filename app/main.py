import joblib 
from fastapi import FastAPI 
from pydantic import BaseModel

app = FastAPI(title="Predictive Maintenance API")
model = joblib.load("model.pkl")

class SensorReading(BaseModel):
    sicaklik: float
    titresim: float
    bakimdan_gecen_saat: float

@app.post("/predict")
def predict(reading: SensorReading):
    tahmin = model.predict([[reading.sicaklik, reading.titresim, reading.bakimdan_gecen_saat]])
    return {"ariza_riski": int(tahmin[0])}