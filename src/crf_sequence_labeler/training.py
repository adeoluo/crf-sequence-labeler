"""Train the CRF by stochastic gradient ascent on the log-likelihood.

The gradient of the conditional log-likelihood for one sentence is the empirical
feature counts (from the gold labels) minus the expected feature counts (under
the model's current distribution). The expectations use the node and edge
marginals computed by forward-backward. An L2 penalty keeps weights bounded.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from crf_sequence_labeler.crf import CRF, _log_sum_exp
from crf_sequence_labeler.features import sentence_features

LabeledSentence = tuple[list[str], list[str]]


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for CRF training."""

    epochs: int = 10
    learning_rate: float = 0.1
    l2: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


def _node_marginals(
    alpha: list[list[float]], beta: list[list[float]], log_z: float
) -> list[list[float]]:
    return [
        [math.exp(alpha[t][k] + beta[t][k] - log_z) for k in range(len(alpha[t]))]
        for t in range(len(alpha))
    ]


class Trainer:
    """Fits a CRF's weights to labeled sentences."""

    def __init__(self, labels: Sequence[str], config: TrainingConfig | None = None) -> None:
        self._labels = tuple(labels)
        self._index = {label: position for position, label in enumerate(self._labels)}
        self._config = config or TrainingConfig()

    def train(self, data: Sequence[LabeledSentence]) -> CRF:
        """Return a CRF fit to ``data`` by gradient ascent on the log-likelihood.

        Sentences with no tokens are skipped. The training order is shuffled each
        epoch with a generator seeded from the configuration, so ``seed`` controls
        the (reproducible) order in which examples drive the online updates.
        """
        crf = CRF(labels=self._labels)
        rate = self._config.learning_rate
        rng = random.Random(self._config.seed)
        examples = [(tokens, gold) for tokens, gold in data if len(tokens) > 0]
        for _ in range(self._config.epochs):
            rng.shuffle(examples)
            for tokens, gold in examples:
                self._update(crf, tokens, gold, rate)
            rate *= 0.9
        return crf

    def _update(self, crf: CRF, tokens: Sequence[str], gold: Sequence[str], rate: float) -> None:
        features = sentence_features(tokens)
        emissions = [crf._emission_scores(step) for step in features]
        alpha = crf._forward(emissions)
        beta = crf._backward(emissions)
        log_z = _log_sum_exp(alpha[-1])
        node = _node_marginals(alpha, beta, log_z)

        # Emission gradient: empirical minus expected, applied with L2 shrinkage.
        for position, step in enumerate(features):
            gold_label = gold[position]
            for feature in step:
                for label_index, label in enumerate(self._labels):
                    empirical = 1.0 if label == gold_label else 0.0
                    expected = node[position][label_index]
                    key = (feature, label)
                    gradient = empirical - expected - self._config.l2 * crf.emission.get(key, 0.0)
                    if gradient != 0.0:
                        crf.emission[key] = crf.emission.get(key, 0.0) + rate * gradient

        # Transition gradient using edge marginals.
        for position in range(1, len(features)):
            edge = self._edge_marginals(crf, emissions, alpha, beta, log_z, position)
            gold_prev, gold_cur = gold[position - 1], gold[position]
            for i, prev_label in enumerate(self._labels):
                for j, cur_label in enumerate(self._labels):
                    empirical = 1.0 if (prev_label == gold_prev and cur_label == gold_cur) else 0.0
                    expected = edge[i][j]
                    key = (prev_label, cur_label)
                    gradient = empirical - expected - self._config.l2 * crf.transition.get(key, 0.0)
                    if gradient != 0.0:
                        crf.transition[key] = crf.transition.get(key, 0.0) + rate * gradient

    def _edge_marginals(
        self,
        crf: CRF,
        emissions: list[list[float]],
        alpha: list[list[float]],
        beta: list[list[float]],
        log_z: float,
        position: int,
    ) -> list[list[float]]:
        n = len(self._labels)
        edge = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                score = (
                    alpha[position - 1][i]
                    + crf._transition_score(self._labels[i], self._labels[j])
                    + emissions[position][j]
                    + beta[position][j]
                )
                edge[i][j] = math.exp(score - log_z)
        return edge
