"""Tests for CRF features, decoding, forward-backward, and training."""

from __future__ import annotations

import pytest

from crf_sequence_labeler import CRF, Trainer, TrainingConfig, token_features
from crf_sequence_labeler.crf import _log_sum_exp


def test_features_include_context_and_shape() -> None:
    features = token_features(["John", "lives", "here"], 0)
    assert "word=john" in features
    assert "shape=Xxxx" in features
    assert "BOS" in features
    assert any(f.startswith("next_word=") for f in features)


def test_log_sum_exp_matches_reference() -> None:
    import math

    values = [0.0, 1.0, 2.0]
    expected = math.log(sum(math.exp(v) for v in values))
    assert _log_sum_exp(values) == pytest.approx(expected)


def _toy_data() -> list[tuple[list[str], list[str]]]:
    # A simple pattern: capitalized tokens are NAME, lowercase are OTHER.
    return [
        (["John", "runs"], ["NAME", "OTHER"]),
        (["Mary", "sleeps"], ["NAME", "OTHER"]),
        (["Peter", "eats"], ["NAME", "OTHER"]),
        (["she", "met", "Alice"], ["OTHER", "OTHER", "NAME"]),
        (["we", "saw", "Bob"], ["OTHER", "OTHER", "NAME"]),
    ]


def test_training_learns_the_pattern() -> None:
    crf = Trainer(["NAME", "OTHER"], TrainingConfig(epochs=20, seed=0)).train(_toy_data())
    assert crf.viterbi(["David", "runs"]) == ["NAME", "OTHER"]
    assert crf.viterbi(["he", "met", "Sarah"]) == ["OTHER", "OTHER", "NAME"]


def test_partition_function_bounds_gold_score() -> None:
    from crf_sequence_labeler.features import sentence_features

    crf = Trainer(["NAME", "OTHER"], TrainingConfig(epochs=5)).train(_toy_data())
    tokens = ["John", "runs"]
    emissions = [crf._emission_scores(step) for step in sentence_features(tokens)]
    # The log partition is at least the score of any single path.
    best = crf.viterbi(tokens)
    path_score = sum(emissions[t][crf.labels.index(best[t])] for t in range(len(best))) + sum(
        crf._transition_score(best[t - 1], best[t]) for t in range(1, len(best))
    )
    assert crf.log_partition(emissions) >= path_score - 1e-6


def test_empty_sentence_decodes_empty() -> None:
    crf = CRF(labels=("A", "B"))
    assert crf.viterbi([]) == []


def test_training_config_validation() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0)
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=0.0)
