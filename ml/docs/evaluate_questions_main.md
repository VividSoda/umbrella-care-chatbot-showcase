# `main` (in `evaluate_questions.py`) — what it does and why

`main()` is the driver that turns `run_patient` (one simulated conversation)
into an actual experiment: it runs every real patient in the dataset through
all three question-selection configs, under a fair cross-validation setup,
and prints a report comparing them.

## The idea

A single `run_patient` call only tells you about one fake patient under one
strategy — not enough to draw a conclusion. `main()` answers the real
question: **across the whole dataset, does a smarter question-selection
strategy (`partial+IG`) actually reach the correct diagnosis faster than the
naive one (`current`/`partial`)?**

To answer that fairly, three things have to be true:
- The model being tested must never have seen the patient it's being tested
  on (otherwise accuracy is inflated by memorization, not real diagnosis).
- Every config must be tested on the *exact same* patients, with the *exact
  same* random "opening symptoms", so the only thing that differs between
  configs is the strategy itself.
- Results need to be aggregated across many patients and summarized by
  "how many questions in" — not just a single final accuracy number — since
  what's interesting is *how fast* each strategy converges.

`main()`'s structure exists to satisfy all three.

## Step by step, mapped to the code

**1. Load the real dataset.**
```python
X, y = load()
```
`X` is the full symptom matrix (every real patient row from the training
data), `y` their true diseases. This is real data, not simulated.

**2. Set up an accumulator per config.**
```python
totals = {c: np.zeros((MAX_Q + 1, 2)) for c in CONFIGS}
patients = 0
```
`totals[config]` is a running sum, shape `(MAX_Q + 1, 2)` — one row per
round of questioning (0 through `MAX_Q`), two columns (top-1 correct,
top-3 correct). Every patient's `run_patient` result gets added into this
sum; `patients` counts how many patients actually contributed, so an
average can be taken at the end.

**3. Split into folds — "never saw test".**
```python
folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)
for fold, (train_idx, test_idx) in enumerate(folds.split(X, y)):
    sim = Sim(BernoulliNB().fit(X[train_idx], y[train_idx]))  # never saw test
```
Instead of using the single pre-trained production model (which has already
seen every patient in training), a *fresh* model is trained from scratch on
only the `train_idx` rows for this fold, then wrapped in `Sim`. Every patient
evaluated in this fold (`test_idx`) is one the model has never encountered —
this is what makes the resulting accuracy numbers trustworthy rather than
optimistic. `StratifiedKFold` also makes sure each fold's train/test split
keeps roughly the same disease proportions as the full dataset, so rare
diseases aren't accidentally missing from training or testing.

**4. Pick which test patients are usable.**
```python
for i in test_idx:
    if X[i].sum() < START:
        continue  # skip patients who don't have enough symptoms to start with
    patients += 1
```
`run_patient` requires `START` symptoms to seed the patient's "opening
statement" (see `run_patient.md`). A patient with fewer true symptoms than
`START` can't be simulated that way, so they're skipped entirely — not
counted toward `patients` and not run through any config.

**5. Run every config on the same patient, fairly.**
```python
for config in CONFIGS:
    rng = np.random.default_rng(1000 * fold + int(i))
    steps = run_patient(sim, X[i], y[i], config, rng)
    totals[config] += np.array(steps, dtype=float)
```
This is the fairness-critical line. A *new* random generator is created for
every `(patient, config)` pair, but seeded from `fold` and the patient index
`i` only — not from the config. That means all three configs, for this same
patient, draw the *identical* random "opening statement" from `run_patient`,
so any difference in the resulting accuracy trajectory can only be due to
the question-selection strategy, not due to one config getting luckier
starting symptoms than another. `run_patient`'s returned list of
`(top1_correct, top3_correct)` booleans gets cast to floats and added
element-wise into that config's running total.

**6. Print the comparison report.**
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
`totals[config][:, col] / patients` turns each config's accumulated sum of
successes at each round into an *average accuracy across all patients* at
that round — e.g. row `[0.41, 0.58, 0.67, ...]` means "41% of patients had
the right top-1 guess with 0 questions asked, 58% after 1 question," etc.
Two such tables are printed — one for top-1 accuracy, one for top-3 — each
with one row per config, so you can read straight across and see which
strategy climbs to high accuracy in fewer questions.

## The end goal

A printed side-by-side table, for top-1 and top-3 accuracy, showing — for
each of `current`, `partial`, and `partial+IG` — what fraction of (held-out,
never-trained-on) patients would have the correct diagnosis after 0, 1, 2,
... up to `MAX_Q` follow-up questions. That's the evidence needed to decide
whether switching the chatbot's question-selection strategy to info-gain is
actually worth it.

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
