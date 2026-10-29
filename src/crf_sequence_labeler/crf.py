"""A linear-chain conditional random field, trained from scratch.

The CRF models ``p(labels | tokens)`` with emission weights (feature, label) and
transition weights (previous label, label). Three algorithms are implemented in
log space for numerical stability:

* **Viterbi** - the most probable label sequence (decoding).
* **Forward-backward** - the log partition function and the marginal
  probabilities needed for the gradient.
* **Log-likelihood gradient** - empirical minus expected feature counts, used by
  stochastic gradient ascent with L2 regularization.

Everything is pure Python; the only numerics are a log-sum-exp helper.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from crf_sequence_labeler.features import sentence_features

EmissionFeatures = list[list[str]]


def _log_sum_exp(values: Sequence[float]) -> float:
    highest = max(values)
    if highest == -math.inf:
        return -math.inf
    return highest + math.log(sum(math.exp(value - highest) for value in values))


@dataclass
class CRF:
    """A linear-chain CRF over a fixed label set."""

    labels: tuple[str, ...]
    emission: dict[tuple[str, str], float] = field(default_factory=dict)
    transition: dict[tuple[str, str], float] = field(default_factory=dict)

    def _emission_scores(self, features: list[str]) -> list[float]:
        return [
            sum(self.emission.get((feature, label), 0.0) for feature in features)
            for label in self.labels
        ]

    def _transition_score(self, previous: str, current: str) -> float:
        return self.transition.get((previous, current), 0.0)

    def viterbi(self, tokens: Sequence[str]) -> list[str]:
        """Return the most probable label sequence for ``tokens``."""
        features = sentence_features(tokens)
        if not features:
            return []
        n_labels = len(self.labels)
        emissions = [self._emission_scores(step) for step in features]
        best = [emissions[0][label] for label in range(n_labels)]
        backpointers: list[list[int]] = []
        for position in range(1, len(features)):
            new_best = [-math.inf] * n_labels
            pointers = [0] * n_labels
            for current in range(n_labels):
                for previous in range(n_labels):
                    score = (
                        best[previous]
                        + self._transition_score(self.labels[previous], self.labels[current])
                        + emissions[position][current]
                    )
                    if score > new_best[current]:
                        new_best[current] = score
                        pointers[current] = previous
            best = new_best
            backpointers.append(pointers)
        last = max(range(n_labels), key=lambda label: best[label])
        path = [last]
        for pointers in reversed(backpointers):
            last = pointers[last]
            path.append(last)
        path.reverse()
        return [self.labels[index] for index in path]

    def _forward(self, emissions: list[list[float]]) -> list[list[float]]:
        n_labels = len(self.labels)
        alpha = [list(emissions[0])]
        for position in range(1, len(emissions)):
            current_alpha = [-math.inf] * n_labels
            for current in range(n_labels):
                scores = [
                    alpha[position - 1][previous]
                    + self._transition_score(self.labels[previous], self.labels[current])
                    for previous in range(n_labels)
                ]
                current_alpha[current] = _log_sum_exp(scores) + emissions[position][current]
            alpha.append(current_alpha)
        return alpha

    def _backward(self, emissions: list[list[float]]) -> list[list[float]]:
        n_labels = len(self.labels)
        length = len(emissions)
        beta = [[-math.inf] * n_labels for _ in range(length)]
        beta[length - 1] = [0.0] * n_labels
        for position in range(length - 2, -1, -1):
            for current in range(n_labels):
                scores = [
                    self._transition_score(self.labels[current], self.labels[nxt])
                    + emissions[position + 1][nxt]
                    + beta[position + 1][nxt]
                    for nxt in range(n_labels)
                ]
                beta[position][current] = _log_sum_exp(scores)
        return beta

    def log_partition(self, emissions: list[list[float]]) -> float:
        """Return the log partition function via the forward pass."""
        alpha = self._forward(emissions)
        return _log_sum_exp(alpha[-1])
