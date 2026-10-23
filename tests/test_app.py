import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.main import app, get_models
from src.features import FEATURES, load
from src.predict import predict_json

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)
EX = {"CRIM": 0.25651, "ZN": 0.0, "INDUS": 9.69, "CHAS": 0, "NOX": 0.538, "RM": 6.2085, "AGE": 77.5,
      "DIS": 3.20745, "RAD": 5, "TAX": 330, "PTRATIO": 19.05, "LSTAT": 11.36}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict():
    b = client.post("/predict", json=EX).json()
    lo, hi = b["quantile_range_5_95"]
    assert 0 < lo < hi and 5 <= b["medv_thousands_usd"] <= 55


def test_b_feature_is_excluded():
    assert "B" not in FEATURES
    assert "B" not in json.loads((ROOT / "web" / "model.json").read_text())["features"]
    assert client.post("/predict", json={**EX, "B": 396.9}).status_code == 200  # extra field ignored


def test_validation():
    assert client.post("/predict", json={**EX, "RM": 20}).status_code == 422


def test_more_rooms_is_not_cheaper():
    small = client.post("/predict", json={**EX, "RM": 5.0}).json()["medv_thousands_usd"]
    large = client.post("/predict", json={**EX, "RM": 8.0}).json()["medv_thousands_usd"]
    assert large > small


def test_json_export_matches_sklearn():
    m = get_models()
    model = json.loads((ROOT / "web" / "model.json").read_text())
    df = load()
    ref = {k: np.exp(m[k].predict(df[FEATURES].to_numpy())) for k in ("point", "lo", "hi")}
    for i, row in enumerate(df[FEATURES].to_dict("records")):
        got = predict_json(model, row)
        for k in got:
            assert abs(got[k] - ref[k][i]) < 1e-6
