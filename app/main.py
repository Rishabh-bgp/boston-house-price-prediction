"""FastAPI service for the Boston house price model (B feature excluded)."""
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import FEATURES

MODEL = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
app = FastAPI(title="Boston House Price API", version="1.0.0")
_models = None


def get_models():
    global _models
    if _models is None:
        if not MODEL.exists():
            from src.train import train
            train()
        _models = joblib.load(MODEL)
    return _models


class House(BaseModel):
    CRIM: float = Field(..., ge=0, le=100, examples=[0.25651], description="Per-capita crime rate by town")
    ZN: float = Field(..., ge=0, le=100, examples=[0.0], description="% residential land zoned for lots > 25,000 sq ft")
    INDUS: float = Field(..., ge=0, le=30, examples=[9.69], description="% non-retail business acres")
    CHAS: int = Field(..., ge=0, le=1, examples=[0], description="1 if tract bounds the Charles River")
    NOX: float = Field(..., ge=0.3, le=0.9, examples=[0.538], description="Nitric oxides concentration (parts per 10 million)")
    RM: float = Field(..., ge=3, le=9, examples=[6.2085], description="Average rooms per dwelling")
    AGE: float = Field(..., ge=0, le=100, examples=[77.5], description="% owner-occupied units built before 1940")
    DIS: float = Field(..., ge=1, le=13, examples=[3.20745], description="Weighted distance to five Boston employment centres")
    RAD: float = Field(..., ge=1, le=24, examples=[5], description="Index of accessibility to radial highways")
    TAX: float = Field(..., ge=150, le=750, examples=[330], description="Property-tax rate per $10,000")
    PTRATIO: float = Field(..., ge=12, le=23, examples=[19.05], description="Pupil-teacher ratio by town")
    LSTAT: float = Field(..., ge=1, le=40, examples=[11.36], description="% lower status of the population")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(h: House):
    d = h.model_dump()
    x = np.array([[d[f] for f in FEATURES]])
    m = get_models()
    point, lo, hi = (float(np.exp(m[k].predict(x)[0])) for k in ("point", "lo", "hi"))
    return {"medv_thousands_usd": round(point, 3), "quantile_range_5_95": [round(lo, 3), round(hi, 3)],
            "unit": "median home value, thousands of 1970s US dollars"}
