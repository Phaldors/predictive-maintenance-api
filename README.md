# Predictive Maintenance API

A small ML-powered API that predicts whether a machine is at risk of failing
within the next 7 days, based on sensor readings (temperature, vibration, and
hours since last maintenance). Ties classical machine learning to a real
electrical-electronics engineering background (industrial maintenance,
fault diagnosis).

## How it works

- `app/data.py` generates a synthetic but realistic sensor dataset (noise
  included) mimicking machine telemetry.
- `app/train.py` trains a `RandomForestClassifier` on that data (~88% accuracy)
  and saves it with `joblib`.
- `app/main.py` is a FastAPI service that loads the trained model and exposes
  a `POST /predict` endpoint.
- `Dockerfile` builds a self-contained image: it installs dependencies,
  trains the model at build time, and serves the API on container start.

## Run it locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app/train.py          # trains and saves model.pkl
uvicorn app.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sicaklik": 95, "titresim": 9, "bakimdan_gecen_saat": 1800}'
```

## Run it with Docker

```bash
docker build -t predictive-maintenance-api .
docker run -p 8000:8000 predictive-maintenance-api
```

## Stack

Python, scikit-learn, FastAPI, joblib, Docker.
