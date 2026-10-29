"""A deterministic micro-benchmark for training and tagging throughput.

Measures CRF training time and Viterbi decoding throughput on the bundled data,
plus the held-out span F1. Deterministic and offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from crf_sequence_labeler.data import LABELS, test_data, train_data
from crf_sequence_labeler.evaluation import evaluate_spans
from crf_sequence_labeler.training import Trainer, TrainingConfig


def main() -> None:
    data = train_data()
    start = time.perf_counter()
    crf = Trainer(LABELS, TrainingConfig(epochs=25, seed=0)).train(data)
    train_ms = (time.perf_counter() - start) * 1000

    report = evaluate_spans(crf, test_data())

    sentences = [tokens for tokens, _ in data] * 50
    start = time.perf_counter()
    for tokens in sentences:
        crf.viterbi(tokens)
    decode_s = time.perf_counter() - start

    print(f"Train sentences: {len(data)} | Labels: {len(LABELS)}")
    print(f"Training (25 epochs): {train_ms:.0f} ms")
    print(f"Held-out span F1: {report.f1:.3f}")
    print(f"  precision {report.precision:.3f}  recall {report.recall:.3f}")
    print(f"Viterbi decoding: {len(sentences) / decode_s:.0f} sentences/sec")


if __name__ == "__main__":
    main()
