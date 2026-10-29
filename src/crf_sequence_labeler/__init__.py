"""CRF Sequence Labeler - a linear-chain conditional random field from scratch."""

from __future__ import annotations

from crf_sequence_labeler.crf import CRF
from crf_sequence_labeler.features import sentence_features, token_features
from crf_sequence_labeler.training import Trainer, TrainingConfig

__version__ = "0.1.0"

__all__ = [
    "CRF",
    "Trainer",
    "TrainingConfig",
    "__version__",
    "sentence_features",
    "token_features",
]
