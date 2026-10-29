"""Span-level evaluation for sequence labeling.

Named-entity quality is measured on whole spans, not individual tags: a predicted
entity counts as correct only if its type and its exact token boundaries match a
gold entity. Precision, recall, and F1 are computed over spans, and compared with
an all-outside baseline that predicts no entities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from crf_sequence_labeler.crf import CRF

Span = tuple[str, int, int]


def extract_spans(tags: Sequence[str]) -> set[Span]:
    """Extract (type, start, end) entity spans from a BIO tag sequence."""
    spans: set[Span] = set()
    current_type: str | None = None
    start = 0
    for index, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_type is not None:
                spans.add((current_type, start, index))
            current_type = tag[2:]
            start = index
        elif tag.startswith("I-") and current_type == tag[2:]:
            continue
        else:
            if current_type is not None:
                spans.add((current_type, start, index))
            current_type = None
    if current_type is not None:
        spans.add((current_type, start, len(tags)))
    return spans


@dataclass(frozen=True)
class SpanReport:
    """Span-level precision, recall, and F1."""

    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def format(self) -> str:
        return f"precision {self.precision:.3f}  recall {self.recall:.3f}  F1 {self.f1:.3f}"


def evaluate_spans(crf: CRF, data: Sequence[tuple[list[str], list[str]]]) -> SpanReport:
    """Score span-level precision, recall, and F1 of the CRF on labeled data."""
    tp = fp = fn = 0
    for tokens, gold in data:
        predicted = crf.viterbi(tokens)
        gold_spans = extract_spans(gold)
        predicted_spans = extract_spans(predicted)
        tp += len(gold_spans & predicted_spans)
        fp += len(predicted_spans - gold_spans)
        fn += len(gold_spans - predicted_spans)
    return SpanReport(true_positive=tp, false_positive=fp, false_negative=fn)
