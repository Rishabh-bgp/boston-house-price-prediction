"""Data loading for the Boston housing data.

The 'B' column is a transformation of the proportion of Black residents per town. It is
excluded from every model in this repository for ethical reasons (see README).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "boston_housing.csv"
TARGET = "MEDV"
EXCLUDED = ["B"]
FEATURES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT"]


def load() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)
