from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pathlib import Path
from joblib import load
import pandas as pd
from .schemas import VectorPayload, BatchPayload

app = FastAPI(title="Rain Forecast API", version="1.0.0")
MODELS_ROOT = Path(__file__).resolve().parents[2] / "models"

def load_bundle(task_folder: str):
    root = MODELS_ROOT / task_folder
    model = load(root / "model.joblib")
    features = (root / "features.txt").read_text().strip().splitlines()
    return model, features

cls_model, cls_feats = load_bundle("rain_or_not")
reg_model, reg_feats = load_bundle("precipitation_fall")

@app.get("/")
def read_root():
    return {"hello": "world", "classification_features": len(cls_feats), "regression_features": len(reg_feats)}

@app.get("/health", status_code=200)
def healthcheck():
    return "Models are ready."

def coerce_df(obs: dict, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{c: float(obs.get(c, 0.0)) for c in columns}])

@app.post("/predict/classification")
def predict_cls(payload: VectorPayload, threshold: float = 0.35):
    X = coerce_df(payload.features, cls_feats)
    prob = float(cls_model.predict_proba(X)[:, 1][0])
    pred = int(prob >= threshold)
    return {"probability": prob, "prediction": pred, "threshold": threshold}

@app.post("/predict/classification/batch")
def predict_cls_batch(payload: BatchPayload, threshold: float = 0.35):
    df = pd.DataFrame(payload.rows)
    X = pd.DataFrame({c: pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0) for c in cls_feats})
    proba = cls_model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return JSONResponse({"probabilities": proba.tolist(), "predictions": pred.tolist(), "threshold": threshold})

@app.post("/predict/regression")
def predict_reg(payload: VectorPayload):
    X = coerce_df(payload.features, reg_feats)
    yhat = float(reg_model.predict(X)[0])
    return {"precip_3d_mm": yhat}

@app.post("/predict/regression/batch")
def predict_reg_batch(payload: BatchPayload):
    df = pd.DataFrame(payload.rows)
    X = pd.DataFrame({c: pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0) for c in reg_feats})
    yhat = reg_model.predict(X)
    return JSONResponse({"precip_3d_mm": yhat.tolist()})

