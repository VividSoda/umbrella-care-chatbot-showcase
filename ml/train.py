"""Trains the model from ml/data and writes artifacts/model.joblib."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB

DATA_PATH = Path(__file__).parent / "data" / "Training.csv"
TARGET = "prognosis"

MODEL_PATH = Path(__file__).parent.parent / "artifacts" / "model.joblib"


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

    model = BernoulliNB()
    model.fit(X_train, y_train)

    print(
        "diseases the model knows:", len(model.classes_)
    )  # model.classes_ is a list of the disease names the model has seen in training

    # try one prediction: a patient with itching and skin rash
    sample = pd.DataFrame([[0] * X.shape[1]], columns=X.columns)
    sample["itching"] = 1
    sample["skin_rash"] = 1

    probs = model.predict_proba(sample)[0]
    for i in probs.argsort()[::-1][:3]:  # indexes of the 3 highest probabilities
        print(f"{model.classes_[i]}: {probs[i]:.2f}")

    # 1. accuracy: fraction of test rows where the top guess is correct
    predictions = model.predict(X_test)
    print("test accuracy:", round(accuracy_score(y_test, predictions), 3))

    # 2. top-3 accuracy: is the true disease among the 3 most likely?
    probs = model.predict_proba(X_test)
    top3 = np.argsort(probs, axis=1)[:, ::-1][:, :3]
    hits = [y_test.iloc[i] in model.classes_[top3[i]] for i in range(len(y_test))]
    print("top-3 accuracy:", round(float(np.mean(hits)), 3))

    # 3. cross-validation: repeat train/test 5 times on different splits
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(BernoulliNB(), X, y, cv=cv)
    print("cross-validation:", scores.round(2), "mean", round(scores.mean(), 3))

    # final model: learn from ALL rows, then save it with the symptom order
    final_model = BernoulliNB().fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final_model, "symptoms": list(X.columns)}, MODEL_PATH)
    print("saved:", MODEL_PATH)

    check_saved_model()


def check_saved_model() -> None:
    bundle = joblib.load(MODEL_PATH)
    model, symptoms = bundle["model"], bundle["symptoms"]

    row = pd.DataFrame([[0] * len(symptoms)], columns=symptoms)
    row["itching"] = 1
    row["skin_rash"] = 1
    print("reloaded model predicts:", model.predict(row)[0])


if __name__ == "__main__":
    main()
