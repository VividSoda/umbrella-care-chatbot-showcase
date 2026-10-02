import numpy as np
from scipy.special import softmax


class Sim:
    def __init__(self, model):
        self.classes = np.array([c.strip() for c in model.classes_])
        self.log_prob_of_diseases = (
            model.class_log_prior
        )  # log probabilities i.e log P(disease)
        self.log_prob_of_symptoms_given_diseases = (
            model.feature_log_prob_
        )  # log P(symptom present | disease)
        self.log_prob_of_symptoms_absent_given_diseases = np.log1p(
            -np.exp(self.log_prob_of_symptoms_given_diseases)
        )  # log P(symptom absent | disease)

        """ derives log P(symptom=0 | disease) from the above, since for a binary symptom P(0) = 1 - P(1). 
        It does this as exp then log1p(-x) rather than just np.log(1 - np.exp(self.lp1)) 
        for numerical stability — log1p(x) computes log(1+x) accurately even when x is very close to 0, 
        which matters when P(1) is very close to 1 (so 1 - P(1) is tiny) — direct subtraction would lose precision there."""

        self.p1 = np.exp(
            self.log_prob_of_symptoms_given_diseases
        )  # P(symptom present | disease)

    def posterior_full(self, indices_of_confirmed_present_symptoms):
        """posterior_full(yes) returns the model's full probability distribution over all diseases,
        given only the symptoms confirmed present so far — treating every symptom not in yes as if it were confirmed absent.
        Concretely: it returns an array of length 41 (one number per disease),
        where each number is P(disease | symptoms), and the whole array sums to 1.
        For example, if yes = {itching, skin_rash}, you might get back something like:
        """
        x = np.zeros(self.p1.shape[1])
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
        scores = self.prior.copy()
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
        top = int(np.argmax(posterior_probability))
        for j in np.argsort(self.p1[top])[::-1]:
            if int(j) not in asked_symptom_indices:
                return int(j)
