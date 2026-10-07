import numpy as np
from utils import entropy, softmax


class Sim:
    def __init__(self, model):
        self.classes = np.array(
            [c.strip() for c in model.classes_]
        )  # shape (41,) — disease names
        self.log_prob_of_diseases = (
            model.class_log_prior_
        )  # shape (41,) — log P(disease)
        self.log_prob_of_symptoms_given_diseases = (
            model.feature_log_prob_
        )  # shape (41, 132) — log P(symptom present | disease)
        self.log_prob_of_symptoms_absent_given_diseases = np.log1p(
            -np.exp(self.log_prob_of_symptoms_given_diseases)
        )  # shape (41, 132) — log P(symptom absent | disease)

        """ derives log P(symptom=0 | disease) from the above, since for a binary symptom P(0) = 1 - P(1).
        It does this as exp then log1p(-x) rather than just np.log(1 - np.exp(self.lp1))
        for numerical stability — log1p(x) computes log(1+x) accurately even when x is very close to 0,
        which matters when P(1) is very close to 1 (so 1 - P(1) is tiny) — direct subtraction would lose precision there."""

        self.probability_of_symptoms_given_diseases = np.exp(
            self.log_prob_of_symptoms_given_diseases
        )  # shape (41, 132) — P(symptom present | disease)

    def posterior_full(self, indices_of_confirmed_present_symptoms):
        """posterior_full(yes) returns the model's full probability distribution over all diseases,
        given only the symptoms confirmed present so far — treating every symptom not in yes as if it were confirmed absent.
        Concretely: it returns an array of length 41 (one number per disease),
        where each number is P(disease | symptoms), and the whole array sums to 1.
        For example, if yes = {itching, skin_rash}, you might get back something like:
        """
        x = np.zeros(self.probability_of_symptoms_given_diseases.shape[1])
        x[list(indices_of_confirmed_present_symptoms)] = 1
        scores = self.log_prob_of_diseases + (
            x * self.log_prob_of_symptoms_given_diseases
            + (1 - x) * self.log_prob_of_symptoms_absent_given_diseases
        ).sum(axis=1)
        return softmax(scores)

    def posterior_partial(
        self, symptom_indices_confirmed_yes, symptom_indices_confirmed_no
    ):
        """Only answered symptoms count; the rest are left out as unknown
        (not assumed absent). See ml/docs/posterior_partial.md for why that's
        mathematically correct, not just a shortcut."""
        scores = self.log_prob_of_diseases.copy()
        if symptom_indices_confirmed_yes:
            scores += self.log_prob_of_symptoms_given_diseases[
                :, list(symptom_indices_confirmed_yes)
            ].sum(axis=1)
        if symptom_indices_confirmed_no:
            scores += self.log_prob_of_symptoms_absent_given_diseases[
                :, list(symptom_indices_confirmed_no)
            ].sum(axis=1)
        return softmax(scores)

    def ask_typical(self, posterior_probability, asked_symptom_indices):
        """Today: the unasked symptom most common in the top disease."""
        index_of_disease_with_highest_posterior = int(np.argmax(posterior_probability))
        for j in np.argsort(
            self.log_prob_of_symptoms_given_diseases[
                index_of_disease_with_highest_posterior
            ]
        )[::-1]:
            if int(j) not in asked_symptom_indices:
                return int(j)

    def ask_info_gain(self, posterior_probability, asked_symptom_indices):
        """The smarter question-selection strategy (vs. ask_typical): checks ALL
        132 candidate symptoms and picks whichever one is expected to reduce
        uncertainty about the diagnosis the most, on average, regardless of
        which disease currently looks most likely."""

        # the probability a random patient (under the current disease belief) would answer yes to this symptom.
        p_yes = (
            posterior_probability @ self.probability_of_symptoms_given_diseases
        )  # shape (41,) @ shape (41, 132) -> shape (132,): one "chance of yes" per candidate symptom,

        after_yes = (
            posterior_probability[:, None] * self.probability_of_symptoms_given_diseases
        )  # shape (41, 132): column j = P(disease d AND symptom j = yes), per disease d, not yet normalized
        after_yes /= after_yes.sum(
            axis=0, keepdims=True
        )  # normalize each column (each symptom j) so it sums to 1 across diseases —
        # column j is now "the posterior over diseases IF the patient says yes to symptom j"

        # same bug again: "answer is no" should use (1 - self.p1), not (1 - log-probability)
        after_no = posterior_probability[:, None] * np.exp(
            1 - self.log_prob_of_symptoms_given_diseases
        )  # shape (41, 132): mirror of after_yes, for "the answer is no"
        after_no /= after_no.sum(
            axis=0, keepdims=True
        )  # normalize the same way -> "posterior over diseases IF the patient says no to symptom j"

        # expected[j] = P(yes to j) * (uncertainty if yes) + P(no to j) * (uncertainty if no)
        expected = p_yes * entropy(after_yes.T) + (1 - p_yes) * entropy(after_no.T)
        gain = entropy(posterior_probability) - expected
        gain[list(asked_symptom_indices)] = -np.inf  # don't ask the same question twice
        return int(np.argmax(gain))
