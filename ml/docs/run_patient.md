# `run_patient` — what it does and why

`run_patient(sim, symptoms, truth, config, rng)` simulates one fake patient
conversation and returns a scorecard of how the model's accuracy evolves as
more symptom questions get asked. It's the core unit of the question-strategy
evaluation: compare these scorecards across strategies (`ask_typical` vs
`ask_info_gain`) over many simulated patients to see which one reaches the
right diagnosis in fewer questions.

## The story, step by step

**1. Invent a patient.**
A fake patient needs a known answer key: their true disease (`truth`) and
their true full set of symptoms (`symptoms`). This is ground truth — what the
model is trying to guess, but isn't allowed to see directly.

**2. Give the patient a random "opening statement".**
In real life, a patient doesn't list every symptom they have the moment they
open the chat — they mention a few. So a small handful of this patient's true
symptoms is randomly picked to count as "already known" before any
question-asking begins. This is why randomness (`rng`) enters at all: it
stands in for "which symptoms a real patient happens to volunteer first,"
which varies patient to patient.

**3. Ask the model: what do you believe right now?**
Using only what's known so far (the opening statement, plus anything learned
in later rounds), the model produces its current best guess — a ranked list
of candidate diseases, most likely first. This belief is recalculated fresh
every round as more information comes in.

**4. Grade that belief.**
Before asking anything new, two yes/no checks are recorded: is the model's
#1 guess right? Is the true disease at least somewhere in its top 3 guesses?
This happens at *every* round, not just the end, because what actually
matters is the trajectory — how quickly accuracy climbs as questions get
asked, not just the final answer.

**5. Pick the next question.**
Given the model's current belief, one of two strategies decides which
yet-unasked symptom is most worth asking about next. "Naive" (`ask_typical`)
just looks at what's typical for whatever disease currently looks most
likely. "Smart" (`ask_info_gain`) looks at all possible symptoms and
estimates which one would most reduce the model's overall uncertainty,
regardless of what it currently suspects.

**6. Consult the answer key honestly.**
The fake patient "answers" the chosen question truthfully, based on their
real symptom list from step 1 — this is the only place the ground truth
leaks into the model's working knowledge, one bit at a time, exactly like a
real conversation.

**7. Repeat.**
Loop back to step 3 with the newly expanded knowledge, for a fixed number of
rounds (`MAX_Q`).

## The end goal

A sequence of accuracy snapshots — one `(top1_correct, top3_correct)` pair
per question asked — for one simulated patient under one strategy. Run this
over many fake patients and compare the snapshot sequences between "naive"
and "smart" strategies to answer: does smarter question selection get you to
the right diagnosis in fewer questions?

## Code

```python
def run_patient(sim, symptoms, truth, config, rng):
    """One simulated chat. Returns (top1_correct, top3_correct) per step."""
    present = np.flatnonzero(symptoms)
    yes = set(rng.choice(present, size=min(START, len(present)), replace=False))
    no = set()
    asked = set(yes)
    steps = []
    for q in range(MAX_Q + 1):
        if config == "current":
            post = sim.posterior_full(yes)
        else:
            post = sim.posterior_partial(yes, no)

        best = sim.classes[np.argsort(post)[::-1]]
        steps.append((best[0] == truth, truth in best[:3]))
        if q == MAX_Q:
            break

        if config == "partial+IG":
            j = sim.ask_info_gain(post, asked)
        else:
            j = sim.ask_typical(post, asked)
        asked.add(j)
        (yes if symptoms[j] else no).add(j)  # the patient answers truthfully
    return steps
```
