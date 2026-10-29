"""CRF Sequence Labeler - a linear-chain conditional random field from scratch."""

from __future__ import annotations

from crf_sequence_labeler.crf import CRF
from crf_sequence_labeler.data import LABELS, test_data, train_data
from crf_sequence_labeler.evaluation import SpanReport, evaluate_spans, extract_spans
from crf_sequence_labeler.features import sentence_features, token_features
from crf_sequence_labeler.training import Trainer, TrainingConfig

__version__ = "0.1.0"

__all__ = [
    "CRF",
    "LABELS",
    "SpanReport",
    "Trainer",
    "TrainingConfig",
    "__version__",
    "evaluate_spans",
    "extract_spans",
    "sentence_features",
    "test_data",
    "token_features",
    "train_data",
]
