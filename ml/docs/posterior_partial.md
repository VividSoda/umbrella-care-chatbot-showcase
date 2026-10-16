# `posterior_partial` — what it does and why

`posterior_partial(yes, no)` returns `P(disease | known symptoms)` for every
disease — the same kind of output as `posterior_full` (an array of 41
probabilities summing to 1) — but built only from symptoms we've actually
asked about.

- `yes` = symptom indices confirmed **present**
- `no` = symptom indices confirmed **absent**
- anything in neither set = genuinely **unknown** (never asked), and is
  simply left out of the calculation instead of being guessed one way or
  the other.

## Why "leave it out" is correct, not a shortcut

For a symptom we don't know the value of, the proper move is to
marginalize it — sum over both possibilities, weighted by their
probability:

```
P(present | disease) + P(absent | disease) = 1
```

Since those two always add up to 1, that symptom's contribution to the
product of probabilities is just a factor of 1 — i.e. `log(1) = 0` when
added to the running log-score. So "add nothing for symptoms we haven't
asked about" isn't an approximation, it's exactly what marginalizing an
unknown binary symptom reduces to.

## Why it's more realistic than `posterior_full`

`posterior_full` silently treats every unasked symptom as "confirmed
absent," which can wrongly count against diseases whose symptoms just
haven't come up yet. `posterior_partial` waits until a symptom is
actually answered before it affects any disease's score at all — which
is what you want when simulating the chatbot mid-conversation, before
all follow-up questions have been asked.
