"""Trains the model from ml/data and writes artifacts/model.joblib."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "data" / "Training.csv"
TARGET = "prognosis"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # 1. drop the empty trailing column ("Unnamed: 133")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # 2. fix stray spaces in a few symptom names: "dischromic _patches" -> "dischromic_patches"
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
    )
