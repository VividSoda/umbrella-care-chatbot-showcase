"""Honest evaluation: how good is the model when the user names only a few symptoms?"""

import numpy as np
from simulator import Sim
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from train import TARGET, load_data

TRIES = 5  # different random symptom picks per patient
BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]


def load():
    df = load_data()
    X = df.drop(columns=TARGET).to_numpy(dtype=int)
    y = df[TARGET].str.strip().to_numpy()
    return X, y


def pick_symptoms(patient_row, k, rng):
    """Pretend the patient only told us k of the symptoms they really have."""
    have = np.flatnonzero(patient_row)  # indexes of the symptoms the patient has
    return set(rng.choice(have, size=min(k, len(have), replace=False)))


def collect_results(X, y, k, method):
    """Returns one row per test: [confidence, 1 if the guess was right else 0]."""
    results = []
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_rows, test_rows in folds.split(X, y):
        sim = Sim(
            BernoulliNB().fit(X[train_rows], y[train_rows])
        )  # never saw the test patients

        for i in test_rows:
            for attempt in range(TRIES):
                rng = np.random.default_rng(i * 100 + attempt)
                told = pick_symptoms(X[i], k, rng)
                if method == "current":
                    beliefs = sim.posterior_full(told)  # unmentioned = absent (today)
                else:
                    beliefs = sim.posterior_partial(
                        told, set()
                    )  # unmentioned = unknown
                guess = sim.classes[beliefs.argmax()]
                results.append((beliefs.max(), guess == y[i]))
    return np.array(results)


def print_accuracy_table(X, y):
    print("k = symptoms the user named\n")
    print(" k    current   partial")
    for k in (2, 3, 5):
        now = collect_results(X, y, k, "current")[:, 1].mean()
        better = collect_results(X, y, k, "partial")[:, 1].mean()
        print(f" {k}    {now:6.1%}    {better:6.1%}")


def print_reliability_table(results):
    confidence, right = results[:, 0], results[:, 1]
    print("said (confidence)   cases   really right   average said")
    for low, high in BINS:
        in_bin = (confidence >= low) & (confidence < high)
        if in_bin.any():
            print(
                f"{low:.1f} - {min(high, 1.0):.1f}        {in_bin.sum():6d}"
                f"   {right[in_bin].mean():10.0%}   {confidence[in_bin].mean():10.0%}"
            )


def print_threshold_table(results):
    confidence, right = results[:, 0], results[:, 1]
    print("threshold   answered   right when answered")
    for t in (0.2, 0.3, 0.4, 0.5, 0.6):
        answered = confidence >= t
        if answered.any():
            print(
                f"{t:.1f}         {answered.mean():6.0%}      {right[answered].mean():10.0%}"
            )


def main():
    X, y = load()
    print_accuracy_table(X, y)
    print("\n--- reliability, 2 symptoms, current method ---")
    print_reliability_table(collect_results(X, y, 2, "current"))
    print("\n--- reliability, 2 symptoms, ignore-unmentioned ---")
    print_reliability_table(collect_results(X, y, 2, "partial"))
    print("\n--- thresholds, 2 symptoms, ignore-unmentioned ---")
    print_threshold_table(collect_results(X, y, 2, "partial"))


if __name__ == "__main__":
    main()
