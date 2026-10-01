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

    # 3. remove repeated rows (4920 -> 304); otherwise the test set leaks into training
    return df.drop_duplicates().reset_index(drop=True)


def main() -> None:
    df = load_data()
    print("rows after dedupe:", len(df))

    X = df.drop(columns=TARGET)  # features: 132 symptom columns
    y = df[TARGET]  # target: 1 prognosis column (the disease name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("train:", X_train.shape, y_train.shape)


if __name__ == "__main__":
    main()
