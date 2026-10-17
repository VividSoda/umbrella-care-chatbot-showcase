# `ml/evaluate_model.py`: every function explained

One file for the whole script. Each function follows the same order: plain
words, purpose, pseudocode, code, line by line, check yourself, caveats. Read
top to bottom; the functions are in the order the script calls them. The plan
these functions serve is in [`docs/model_evaluation.md`](../../docs/model_evaluation.md).

Run it from the `ml/` folder: `PYTHONPATH=. python3 evaluate_model.py`

## Contents
1. [`load`](#1-load)
2. [`pick_symptoms`](#2-pick_symptoms)
3. [`collect_results`](#3-collect_results)
4. [`print_accuracy_table`](#4-print_accuracy_table)
5. [`print_reliability_table`](#5-print_reliability_table)
6. [`print_threshold_table`](#6-print_threshold_table)
7. [`main`](#7-main)

---

## 1. load

### In plain words

Think of the dataset as a big spreadsheet: one row per patient, one column per
symptom (1 = has it, 0 = does not), plus one last column with the patient's
disease. `load` cuts that spreadsheet in two: the **questions** (symptoms) and
the **answers** (diseases), and hands both over as plain number tables.

### Purpose

Every later function needs the same two things: `X`, what we tell the model, and
`y`, what the right answer was. Doing the cutting once, in one place, means no
function can accidentally leave the answer column inside the symptoms, which
would let the model cheat. It also removes stray spaces from disease names so
`"Acne "` and `"Acne"` are not treated as two diseases.

### Pseudocode

```
output: X (patients x symptoms, 0/1), y (true disease per patient)
df = load the cleaned dataset (train.py already drops duplicate rows)
X = df without the disease column, as whole numbers
y = the disease column, with spaces trimmed off each name
return X, y
```

### Code

```python
def load():
    df = load_data()
    X = df.drop(columns=TARGET).to_numpy(dtype=int)  # 304 patients x 132 symptoms
    y = df[TARGET].str.strip().to_numpy()  # the true disease of each patient
    return X, y
```

### Line by line

- `load_data()` comes from `train.py`, so training and evaluation read the data
  the same way.
- `df.drop(columns=TARGET)` returns the table **without** the disease column. It
  does not change `df`.
- `.to_numpy(dtype=int)` turns a pandas table into a plain numpy grid. Numpy is
  faster and is what scikit-learn expects.
- `df[TARGET].str.strip()` trims spaces from every disease name.

### Check yourself

```python
X, y = load()
print(X.shape, y.shape, y[:2])
```

Expected: `(304, 132) (304,) ['Fungal infection' 'Fungal infection']`

### Caveats

- 304, not 4,920: duplicate rows are already removed. The original data is
  synthetic, so it is a small, repetitive sample.
- Rows are ordered by disease, so never split by position (first 80% / last
  20%). `StratifiedKFold` in `collect_results` handles that properly.

---

## 2. pick_symptoms

### In plain words

A real patient has, say, eight symptoms, but in a chat they type two and wait.
`pick_symptoms` plays that patient. It looks at the symptoms the patient truly
has, pulls out `k` of them at random, and says "this is all they told us".

It is like handing a doctor only two of eight clues and asking for a guess.

### Purpose

The headline accuracy (about 98%) is measured with all 132 symptom columns
filled in. A chat user never gives that much, so that number says little about
the app. `pick_symptoms` is what makes the evaluation honest: it hides most of
the information, the way a real conversation does. `collect_results` calls it
for every patient, every try.

It takes `rng` (a random generator) as an input instead of making its own, so
the caller controls the seed. That lets two methods be tested on **exactly the
same** picks.

### Pseudocode

```
inputs: patient_row (0/1 per symptom), k (how many to reveal), rng
have = the positions where the patient has a 1
how_many = k, but never more than len(have)       # cannot reveal more than they have
told = how_many positions chosen at random from have, no repeats
return told as a set
```

### Code

```python
def pick_symptoms(patient_row, k, rng):
    """Pretend the patient only told us k of the symptoms they really have."""
    have = np.flatnonzero(patient_row)  # positions of the symptoms the patient has
    return set(rng.choice(have, size=min(k, len(have)), replace=False))
```

### Line by line

- `np.flatnonzero(patient_row)`: positions of the non-zero entries. For
  `[0, 1, 0, 1, 1]` it gives `[1, 3, 4]`.
- `min(k, len(have))`: some patients have fewer than `k` symptoms. Asking numpy
  for more than exist raises an error, so we cap it.
- `rng.choice(have, size=..., replace=False)`: pick that many, `replace=False`
  meaning the same symptom cannot be picked twice.
- `set(...)`: `Sim.posterior_full` and `posterior_partial` want a set of symptom
  positions.
- **Mind the brackets.** `size=min(k, len(have))` closes the `min(...)` before
  `, replace=False`. Writing `min(k, len(have), replace=False)` passes
  `replace` to `min`, which fails with a `TypeError`.

### Check yourself

```python
row = np.array([0, 1, 0, 1, 1])
print(pick_symptoms(row, 2, np.random.default_rng(0)))
print(pick_symptoms(row, 9, np.random.default_rng(0)))
```

Real output: `{3, 4}` then `{1, 3, 4}` (the second one asked for 9 but the
patient only has 3, so all 3 come back). Numpy may print them as
`np.int64(3)`; that is the same number.

### Caveats

- It picks uniformly at random. Real users mention their most noticeable
  symptom first, so real chats may be easier or harder than this test.
- Same `rng` seed gives the same picks. That is a feature here, not a bug.

---

## 3. collect_results

`collect_results(X, y, k, method)` is the data-gathering step of the model
evaluation. It pretends to be many users who each name only `k` symptoms, asks
the model for its best guess, and records two numbers for every pretend chat:
how confident the model was, and whether the guess was right. Everything else in
`evaluate_model.py` (accuracy table, reliability table, threshold table) is just
a summary of the rows this function returns.

### The idea

The headline accuracy (~98%) comes from giving the model all 132 symptoms of a
patient. A chat user names two or three. To measure what users actually get, the
test has to:

1. **Hide most of each patient's symptoms**, by keeping only `k` real ones
   (`pick_symptoms`, see `evaluate_questions_main.md`).
2. **Never test on a patient the model trained on**, or accuracy is inflated by
   memorization.
3. **Compare the two belief methods on identical inputs**, so any difference in
   accuracy comes from the method and not from luck.
4. **Keep the confidence next to correctness**, because the later calibration
   checks need both ("when it said 70%, was it right 70% of the time?").

### The two methods

| `method`    | Belief used                                   | Meaning of an unmentioned symptom |
|-------------|-----------------------------------------------|-----------------------------------|
| `"current"` | `sim.posterior_full(told)`                    | absent (what `Predictor.predict` does today) |
| anything else (the tables use `"partial"`) | `sim.posterior_partial(told, set())` | unknown, left out of the calculation |

`posterior_partial(told, set())` passes an empty "confirmed absent" set because
the user only volunteered symptoms they have. Nothing was asked and answered
"no". See `posterior_partial.md` for why leaving unknowns out is the correct
treatment and not a shortcut.

### Pseudo code

```
results = empty list
split patients into 5 folds (same split every run)
for each fold:
    train a fresh model on the other 4 folds        # it has never seen this fold
    for each patient in this fold:
        for each of TRIES attempts:
            seed a random generator from (patient, attempt)
            told = k randomly chosen real symptoms of the patient
            beliefs = chosen method's probabilities for every disease
            guess = disease with the highest probability
            save (highest probability, guess == true disease)
return results as an array, one row per saved pair
```

### Step by step, mapped to the code

#### 1. Cross-validation, so the model never saw the test patient

```python
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_rows, test_rows in folds.split(X, y):
    sim = Sim(BernoulliNB().fit(X[train_rows], y[train_rows]))
```

A new `BernoulliNB` is trained on four folds and wrapped in `Sim`. Only patients
in the held-out fold are tested with it. `StratifiedKFold` keeps disease
proportions similar in each fold, and `random_state=0` makes the split the same
every run.

#### 2. Several random symptom picks per patient

```python
for attempt in range(TRIES):
    rng = np.random.default_rng(i * 100 + attempt)
    told = pick_symptoms(X[i], k, rng)
```

Which `k` symptoms a user mentions is random, and one unlucky pick can mislead.
Repeating `TRIES` times per patient averages that out.

The seed `i * 100 + attempt` depends only on the patient and the attempt, never
on `method`. So `"current"` and `"partial"` are shown **exactly the same**
symptoms for every patient and attempt. The seeds are also unique per pair as
long as `TRIES < 100`.

#### 3. Get beliefs, guess, and record

```python
if method == "current":
    beliefs = sim.posterior_full(told)
else:
    beliefs = sim.posterior_partial(told, set())
guess = sim.classes[beliefs.argmax()]
results.append((beliefs.max(), guess == y[i]))
```

- `beliefs` is one probability per disease, summing to 1.
- `beliefs.argmax()` is the index of the most likely disease and `sim.classes`
  turns it into a disease name.
- `beliefs.max()` is the model's **confidence** in that guess.
- `guess == y[i]` is whether it was **right**.

#### 4. Return a plain array

```python
return np.array(results, dtype=float)
```

Shape `(n_rows, 2)`: column 0 is confidence (0 to 1), column 1 is 1.0 if right
else 0.0. `dtype=float` turns the booleans into 0/1 so they can be averaged.
Number of rows is `patients x TRIES`.

### How the rest of the script uses it

| Caller                           | Uses                         | Answers |
|----------------------------------|------------------------------|---------|
| `print_accuracy_table`           | `results[:, 1].mean()`       | How often is the guess right at `k` = 2, 3, 5? |
| `print_reliability_table`        | both columns, grouped by `BINS` | Is "70% sure" right about 70% of the time? |
| threshold table (step 4)         | both columns, filtered by confidence | At cutoff `t`, how often does the bot answer and how often is it right? |

### Caveats

- The dataset is synthetic with only 304 unique rows (5 to 10 per disease), so
  absolute numbers describe this data, not real patients. The gap between
  `"current"` and `"partial"` is the meaningful part.
- The rows from the `TRIES` attempts of one patient are correlated (same
  patient, same model), so treat them as repeated measurements rather than
  independent samples.
- The `method` argument only checks `== "current"`. Any other string, including
  a typo, silently uses the partial method.

### Code

```python
def collect_results(X, y, k, method):
    """Returns one row per test: [confidence, 1 if the guess was right else 0]."""
    results = []
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_rows, test_rows in folds.split(X, y):
        sim = Sim(BernoulliNB().fit(X[train_rows], y[train_rows]))  # never saw the test patients
        for i in test_rows:
            for attempt in range(TRIES):
                rng = np.random.default_rng(i * 100 + attempt)
                told = pick_symptoms(X[i], k, rng)
                if method == "current":
                    beliefs = sim.posterior_full(told)  # unmentioned = absent (today)
                else:
                    beliefs = sim.posterior_partial(told, set())  # unmentioned = unknown
                guess = sim.classes[beliefs.argmax()]
                results.append((beliefs.max(), guess == y[i]))
    return np.array(results, dtype=float)
```

---

## 4. print_accuracy_table

### In plain words

It prints a small scoreboard: "if the user names 2 symptoms, how often is the
bot's top guess right? What about 3? What about 5?", side by side for today's
behaviour and the fixed behaviour.

### Purpose

This is the table that shows **why the fix is needed**. Today the app treats
every symptom the user did not mention as "they do not have it". The fix treats
those as "we do not know". If the second column is clearly higher, the change to
`Predictor.predict` (Step 3) is justified by numbers, not a hunch.

It only needs `X` and `y`; it asks `collect_results` to do the heavy work and
keeps just one thing from each run: the share of right guesses.

### Pseudocode

```
print a heading and the two column names
for k in (2, 3, 5):
    run the test with the "current" method, take the average of the right/wrong column
    run the test with the "partial" method, take the same average
    print k and both averages as percentages
```

### Code

```python
def print_accuracy_table(X, y):
    print("k = symptoms the user named\n")
    print(" k    current   ignore-unmentioned")
    for k in (2, 3, 5):
        now = collect_results(X, y, k, "current")[:, 1].mean()
        better = collect_results(X, y, k, "partial")[:, 1].mean()
        print(f" {k}    {now:6.1%}    {better:6.1%}")
```

### Line by line

- `collect_results(...)` returns a grid with two columns: confidence and
  right-or-wrong (1.0 or 0.0).
- `[:, 1]` means "every row, column 1", the right/wrong column.
- `.mean()` of 1s and 0s is the **share of ones**, which is the accuracy.
  Three rights out of four tries is 0.75.
- `{now:6.1%}` formats the number as a percentage with one decimal, padded to
  width 6 so the columns line up. `0.436` prints as ` 43.6%`.

### Check yourself

Run `python3 evaluate_model.py` from `ml/` once `main` exists. Real numbers:

```
 k    current   ignore-unmentioned
 2     43.6%     69.1%
 3     62.5%     83.8%
 5     80.7%     93.2%
```

Read it as: with two symptoms, today's logic is right 44% of the time, the fix
69%.

### Caveats

- Each row costs a full cross-validation, so the table runs the test six times.
  It takes a short while; that is expected.
- Synthetic data: trust the **gap** between the columns, not the exact
  percentages.

---

## 5. print_reliability_table

### In plain words

When the bot says "I'm 70% sure", is it right about 70% of the time? A weather
forecaster who says "70% rain" should see rain on about 7 of every 10 such days.
This function checks the bot the same way: sort every test into buckets by how
sure the bot said it was, then count how often it was actually right in each
bucket.

### Purpose

`MIN_CONFIDENCE` in `constants.py` decides when the bot is sure enough to name a
disease, and it was a guess (0.3). Picking a sensible value requires knowing
whether the confidence number means anything. This table answers that. If the
"really right" column is **below** "average said", the bot is overconfident. If
it is **above**, it is underconfident.

### Pseudocode

```
input: results (column 0 = confidence, column 1 = 1 if right else 0)
print column headings
for each bucket (low, high) in BINS:
    pick the rows whose confidence is >= low and < high
    if the bucket is not empty:
        print the bucket range, how many rows, the share that were right,
        and the average confidence the bot claimed
```

### Code

```python
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
```

### Line by line

- `confidence, right = results[:, 0], results[:, 1]` splits the grid into its two
  columns so the rest reads like English.
- `(confidence >= low) & (confidence < high)` is a **mask**: a True/False per
  row, True when the row falls in the bucket. Use `&`, not `and`; `and` fails on
  arrays.
- `in_bin.any()` skips empty buckets, avoiding an average of nothing.
- `in_bin.sum()` counts the Trues, which is the number of cases.
- `right[in_bin].mean()` keeps only the bucket's rows and averages the 1/0
  values: the true hit rate.
- The last bucket is `(0.9, 1.01)` so a confidence of exactly 1.0 is included
  (`< high` would leave it out of `(0.9, 1.0)`). `min(high, 1.0)` just prints
  `1.0` instead of `1.01`.

### Check yourself

```python
res = np.array([[0.9, 1], [0.8, 1], [0.35, 0], [0.25, 0], [0.6, 1.0]])
print_reliability_table(res)
```

Real output, one case per bucket:

```
said (confidence)   cases   really right   average said
0.0 - 0.3             1           0%          25%
0.3 - 0.5             1           0%          35%
0.5 - 0.7             1         100%          60%
0.7 - 0.9             1         100%          80%
0.9 - 1.0             1         100%          90%
```

### What the real run shows (2 symptoms)

Current method, overconfident: it says 82% and is right 65%; says 60%, right
45%.

Ignore-unmentioned method, the opposite: in the 0.3 to 0.5 bucket it says 38%
and is right 92%. Its numbers are too low, but they **rank** cases well, which is
all a threshold needs.

### Caveats

- Buckets with few cases (under about 30) are noisy. Look at `cases` before
  trusting a percentage.
- The `BINS` edges are a choice. Different edges give a slightly different
  picture.

---

## 6. print_threshold_table

### In plain words

The bot needs a rule: "only name a disease if I am at least X% sure; otherwise
ask another question." This table tries several values of X and shows the price
of each: a strict X means the bot rarely answers but is usually right, and a
loose X means it answers often but makes more mistakes.

### Purpose

This is the table you read to **choose `MIN_CONFIDENCE`**. It turns "is 0.3 a
good number?" into a trade-off you can see. The choice is yours: pick the lowest
threshold at which "right when answered" is high enough for you to be
comfortable.

### Pseudocode

```
input: results (column 0 = confidence, column 1 = 1 if right else 0)
print column headings
for each candidate threshold t in (0.2, 0.3, 0.4, 0.5, 0.6):
    answered = the rows where confidence >= t       # the bot would have spoken up
    if there are any:
        print t, the share of all rows that were answered,
        and the share of those answered rows that were right
```

### Code

```python
def print_threshold_table(results):
    confidence, right = results[:, 0], results[:, 1]
    print("threshold   answered   right when answered")
    for t in (0.2, 0.3, 0.4, 0.5, 0.6):
        answered = confidence >= t
        if answered.any():
            print(f"{t:.1f}         {answered.mean():6.0%}      {right[answered].mean():10.0%}")
```

### Line by line

- `answered = confidence >= t` is a True/False mask, True when the bot would have
  answered at this threshold.
- `answered.mean()`: True counts as 1, so the average of a mask is the **share**
  of Trues, here "how often it answers".
- `right[answered].mean()`: keep only the answered rows, then the share that were
  right.
- `if answered.any()` avoids averaging an empty selection at a very high
  threshold.

### Check yourself

```python
res = np.array([[0.9, 1], [0.8, 1], [0.35, 0], [0.25, 0], [0.6, 1.0]])
print_threshold_table(res)
```

Real output:

```
threshold   answered   right when answered
0.2           100%             60%
0.3            80%             75%
0.4            60%            100%
0.5            60%            100%
0.6            60%            100%
```

### What the real run shows (2 symptoms, ignore-unmentioned)

```
threshold   answered   right when answered
0.2            78%             81%
0.3            47%             93%
0.4            22%             99%
0.5             6%            100%
```

At 0.3 the bot answers about half the time and is right 93% of those. At 0.5 it
is perfect but answers 6% of the time, so it would nearly always ask more
questions.

### Caveats

- "Right when answered" is on synthetic data. It is a comparison tool, not a
  promise about real patients.
- A raised threshold never makes the bot less safe: when it does not answer, it
  asks. Emergency and urgency rules do not depend on this threshold.

---

## 7. main

### In plain words

`main` is the "press play" button. Every other function is a tool; `main` picks
them up in the right order, runs the experiment and prints the four tables you
read afterwards.

### Purpose

It makes the evaluation repeatable: one command, same output every time (the
seeds are fixed). Without it you would call the functions by hand in a Python
prompt and could not easily compare a run today with a run after changing the
model.

The `if __name__ == "__main__":` line at the bottom means "only run `main` when
this file is run directly". It stays quiet when something else imports
`collect_results` or `load`.

### Pseudocode

```
X, y = load the data
print the accuracy table (k = 2, 3, 5; current vs ignore-unmentioned)
print the reliability table for 2 symptoms, current method
print the reliability table for 2 symptoms, ignore-unmentioned method
print the threshold table for 2 symptoms, ignore-unmentioned method
```

### Code

```python
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
```

### Line by line

- Two symptoms is used for the last three tables because it is the hardest
  realistic case (the minimum the chat accepts), so a threshold that works there
  works later in the conversation too.
- The threshold table uses `"partial"` because that is the method Step 3 puts
  into the live bot.

### Check yourself

From the `ml/` folder:

```
PYTHONPATH=. python3 evaluate_model.py
```

You should see the accuracy table first (`43.6%` and `69.1%` for k = 2), then
the three tables. If numbers differ, compare `TRIES`, `BINS`, the
`random_state=0` of `StratifiedKFold` and the seed `i * 100 + attempt`; all
must match for identical results.

### Caveats

- It trains 5 models per `collect_results` call and calls it 9 times, so allow
  a little time.
- Synthetic data (304 unique rows): the printed numbers are for comparing
  methods, not for quoting as real-world accuracy.

---
