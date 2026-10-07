"""Compare follow-up question strategies by simulating patients."""

import joblib
import numpy as np
from simulator import Sim
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from train import (  # reuse the cleaning train.py already does
    MODEL_PATH,
    TARGET,
    load_data,
)

START = 2  # symptoms the simulated patient mentions in the first message
MAX_Q = 8  # follow-up questions to simulate
FOLDS = 3
CONFIGS = ["current", "partial", "partial+IG"]

model = joblib.load(MODEL_PATH)
sim = Sim(model)

rng = np.random.default_rng(seed=42)  # for reproducibility


def load():
    df = load_data()
    X = df.drop(columns=TARGET).to_numpy(
        dtype=int
    )  # convert to numpy 2D array and force every value to be 0 or 1
    y = df[TARGET].str.strip().to_numpy()  # the true disease of each patient
    return X, y


def entropy(p):
    """How uncertain a probability distribution is (0 means certain)."""
    s = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(axis=1)


def run_patient(sim, symptoms, truth, config, rng):
    """One simulated chat. Returns (top1_correct, top3_correct) per step."""
    present = np.flatnonzero(symptoms)  # indexes of the symptoms the patient has
    yes = set(rng.choice(present, size=min(START, len(present)), replace=False))
    no = set()  # symptoms confirmed absent so far (none yet)
    asked = set(yes)
    steps = []  # will collect one accuracy result per round of the conversation.
    for q in range(MAX_Q + 1):
        if config == "current":
            post = sim.posterior_full(yes)
        else:
            post = sim.posterior_partial(yes, no)

        best = sim.classes[np.argsort(post)[::-1]]
        steps.append((best[0] == truth, truth in best[:3]))
        if q == MAX_Q:
            break  # don't ask any more questions

        if config == "partial+IG":
            j = sim.ask_info_gain(post, asked)
        else:
            j = sim.ask_typical(post, asked)
        asked.add(j)
        (yes if symptoms[j] else no).add(j)

    return steps


def main():
    X, y = load()
    totals = {
        c: np.zeros((MAX_Q + 1, 2)) for c in CONFIGS
    }  # shape (MAX_Q+1, 2) for top1 and top3 accuracy

    patients = 0

    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)
    for fold, (train_idx, test_idx) in enumerate(folds.split(X, y)):
        sim = Sim(BernoulliNB().fit(X[train_idx], y[train_idx]))  # never saw test

        for i in test_idx:
            if X[i].sum() < START:
                continue  # skip patients who don't have enough symptoms to start with
