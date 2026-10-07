# `main` (in `evaluate_questions.py`) — what it does and why

`main()` is the driver that turns `run_patient` (one simulated conversation)
into an actual experiment. It runs every real patient in the dataset through all
three question-selection strategies, under a fair cross-validation setup, and
prints a report comparing them.

## The idea

A single `run_patient` call tells you about one fake patient under one strategy,
which is not enough to conclude anything. `main()` answers the real question:

> Across the whole dataset, does a smarter question-selection strategy
> (`partial+IG`) reach the correct diagnosis in fewer questions than the naive
> ones (`current` / `partial`)?

To answer that fairly, three things have to be true:

1. **The model never saw the patient it is tested on.** Otherwise accuracy is
   inflated by memorization, not real diagnosis.
2. **Every strategy is tested on the exact same patients with the exact same
   opening symptoms.** Then the only difference between runs is the strategy.
3. **Results are summarized by "questions asked so far"**, not as one final
   number, because what matters is *how fast* each strategy converges.

The structure of `main()` exists to satisfy these three points.

## The three strategies being compared

| Name         | Posterior (what the bot believes)                         | Next question                          |
|--------------|-----------------------------------------------------------|----------------------------------------|
| `current`    | `posterior_full`: unasked symptoms count as absent        | `ask_typical`: most common in top guess |
| `partial`    | `posterior_partial`: unasked symptoms are left as unknown | `ask_typical`                          |
| `partial+IG` | `posterior_partial`                                       | `ask_info_gain`: best at separating candidates |

Comparing `current` to `partial` shows the effect of fixing the belief.
Comparing `partial` to `partial+IG` shows the effect of choosing better
questions.

## Pseudo code

```
load all real patients (symptoms X, true disease y)
make an empty score sheet per strategy (rows = questions asked, cols = top1/top3)

split the patients into 3 folds
for each fold:
    train a fresh model on the other folds          # it has never seen this fold
    for each patient in this fold:
        skip them if they have fewer than 2 symptoms # cannot make an opening statement
        count the patient
        for each strategy:
            reseed the random generator              # same opening symptoms for all strategies
            simulate the chat -> right/wrong at each question number
            add the result into that strategy's score sheet

divide every score sheet by the number of patients   # sums become percentages
print two tables: top-1 accuracy and top-3 accuracy
```

## Step by step, mapped to the code

### 1. Load the real dataset

Real patients with known diseases play the role of the user.

```python
X, y = load()
```

`X` is the full 0/1 symptom matrix (one row per patient). `y` holds their true
diseases. This is real data, not simulated.

### 2. Set up a score sheet per strategy

Each strategy needs a running total that is averaged at the end.

```python
totals = {c: np.zeros((MAX_Q + 1, 2)) for c in CONFIGS}
patients = 0
```

`totals[config]` has shape `(MAX_Q + 1, 2)`:

- one row per moment in the chat, from "0 questions asked" to "`MAX_Q` asked"
- two columns: top-1 correct and top-3 correct

Every patient's result is added into this sum. `patients` counts how many
patients contributed, so an average can be taken later.

### 3. Split into folds, so the model never saw the test patient

```python
folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)
for fold, (train_idx, test_idx) in enumerate(folds.split(X, y)):
    sim = Sim(BernoulliNB().fit(X[train_idx], y[train_idx]))  # never saw test
```

The production model (`model.joblib`) has already trained on every patient, so
testing it on them would be cheating. Instead, for each fold a fresh model is
trained on only the `train_idx` rows, then wrapped in `Sim`. Every patient in
`test_idx` is one that model has never encountered, which makes the accuracy
numbers trustworthy.

`StratifiedKFold` keeps roughly the same disease proportions in every fold, so a
rare disease is not accidentally missing from training or testing.

### 4. Skip patients who cannot be simulated

```python
for i in test_idx:
    if X[i].sum() < START:
        continue  # not enough symptoms for an opening statement
    patients += 1
```

`run_patient` makes the simulated user open with `START` of their real symptoms
(see `run_patient.md`). A patient with fewer symptoms than that cannot be
simulated, so they are skipped and not counted. `patients += 1` sits before the
strategy loop so each patient is counted once, not three times.

### 5. Run every strategy on the same patient, fairly

This is the fairness-critical step.

```python
for config in CONFIGS:
    rng = np.random.default_rng(1000 * fold + int(i))
    steps = run_patient(sim, X[i], y[i], config, rng)
    totals[config] += np.array(steps, dtype=float)
```

- A **new** random generator is created for each `(patient, config)` pair, but
  seeded from `fold` and the patient index `i` only, never from the config.
  All three strategies therefore draw the *identical* random opening symptoms
  for this patient. If one strategy got easier opening symptoms, it would look
  better by luck.
- `run_patient` returns a list of `(top1_correct, top3_correct)` booleans, one per
  question number. Casting to floats and adding element-wise puts them into
  that strategy's score sheet.

### 6. Turn sums into percentages and print

```python
print(f"patients: {patients}, diseases: {len(set(y))}\n")
header = "questions asked:    " + "".join(f"{q:>6}" for q in range(MAX_Q + 1))
for name, col in (("top-1 accuracy", 0), ("top-3 accuracy", 1)):
    print(name)
    print(header)
    for config in CONFIGS:
        row = totals[config][:, col] / patients
        print(f"  {config:<16}" + "".join(f"{v:>6.0%}" for v in row))
    print()
```

`totals[config][:, col] / patients` converts "number of patients right at this
point" into "fraction of patients right at this point". Two tables are printed,
one for top-1 and one for top-3, each with one row per strategy.

## How to read the output

Each row is one strategy. Read across it to see how accuracy climbs as questions
are asked. Illustrative numbers:

```
questions asked:       0     1     2     3
  current            41%   52%   60%   66%
  partial            41%   58%   67%   73%
  partial+IG         41%   70%   85%   92%
```

- All strategies start equal at 0 questions, because nothing has been asked yet.
  If they do not, the fairness setup is broken.
- The strategy whose row climbs fastest converges in the fewest questions.
- Only the *gaps between rows* matter. The dataset has just 304 unique rows, so
  absolute accuracy is inflated; see roadmap item 5.

## The end goal

A side-by-side table showing, for `current`, `partial` and `partial+IG`, what
fraction of held-out patients have the correct diagnosis after 0, 1, 2, ...
`MAX_Q` follow-up questions. That is the evidence for deciding whether the
chatbot should switch to information-gain question selection (roadmap item 1).

## Code

```python
def main():
    X, y = load()
    totals = {c: np.zeros((MAX_Q + 1, 2)) for c in CONFIGS}
    patients = 0

    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)
    for fold, (train_idx, test_idx) in enumerate(folds.split(X, y)):
        sim = Sim(BernoulliNB().fit(X[train_idx], y[train_idx]))  # never saw test
        for i in test_idx:
            if X[i].sum() < START:
                continue
            patients += 1
            for config in CONFIGS:
                rng = np.random.default_rng(1000 * fold + int(i))
                steps = run_patient(sim, X[i], y[i], config, rng)
                totals[config] += np.array(steps, dtype=float)

    print(f"patients: {patients}, diseases: {len(set(y))}\n")
    header = "questions asked:    " + "".join(f"{q:>6}" for q in range(MAX_Q + 1))
    for name, col in (("top-1 accuracy", 0), ("top-3 accuracy", 1)):
        print(name)
        print(header)
        for config in CONFIGS:
            row = totals[config][:, col] / patients
            print(f"  {config:<16}" + "".join(f"{v:>6.0%}" for v in row))
        print()
```
