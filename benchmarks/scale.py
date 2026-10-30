"""A deterministic scale benchmark for CRF Viterbi decoding throughput.

Trains the CRF once on the bundled sentences, then runs Viterbi decoding over
progressively larger sets of synthetic BIO-tagged sentences generated here with
a seeded ``random.Random``. For each workload it reports wall-clock seconds,
decoding throughput in sentences per second, peak memory in MiB measured with
the standard-library ``tracemalloc`` module, and span-level F1 as the quality
metric. Timing is measured without tracemalloc (which otherwise slows every
allocation); peak memory is traced separately around the workload plus a single
decode, which is faithful because decoding processes one sentence at a time.

The sentences are synthetic and follow a small set of templates. This pure-Python
from-scratch implementation is not competitive on wall-clock time with compiled
sequence-labeling libraries; the goal is to show how hand-written Viterbi scales
with the number of sentences. Deterministic and offline; run with:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import random
import time
import tracemalloc
from dataclasses import dataclass

from crf_sequence_labeler.crf import CRF
from crf_sequence_labeler.data import LABELS, train_data
from crf_sequence_labeler.evaluation import evaluate_spans
from crf_sequence_labeler.training import Trainer, TrainingConfig

WORKLOAD_SIZES: tuple[int, ...] = (2000, 4000, 8000, 16000)
_BYTES_PER_MIB = 1024 * 1024

_PERSONS: tuple[str, ...] = (
    "Alice",
    "Bob",
    "Carol",
    "David",
    "Emma",
    "Frank",
    "Grace",
    "Henry",
    "Julia",
    "Kevin",
    "Laura",
    "Mike",
    "Nina",
    "Oscar",
    "Paula",
    "Quentin",
)
_SURNAMES: tuple[str, ...] = ("Smith", "Jones", "Adams", "Clark", "Brown", "Lee")
_CITIES: tuple[str, ...] = (
    "Paris",
    "London",
    "Berlin",
    "Rome",
    "Tokyo",
    "Madrid",
    "Dublin",
    "Oslo",
    "Lisbon",
    "Vienna",
    "Chicago",
    "Boston",
    "Denver",
    "Seattle",
    "Austin",
)
_MULTI_CITIES: tuple[tuple[str, str], ...] = (
    ("New", "York"),
    ("San", "Diego"),
    ("Los", "Angeles"),
    ("Hong", "Kong"),
)

Sentence = tuple[list[str], list[str]]


def _generate(count: int, *, seed: int = 0) -> list[Sentence]:
    """Return ``count`` synthetic BIO-tagged sentences built from templates."""

    rng = random.Random(seed)
    out: list[Sentence] = []
    for _ in range(count):
        person = rng.choice(_PERSONS)
        city = rng.choice(_CITIES)
        multi = rng.choice(_MULTI_CITIES)
        template = rng.randint(0, 5)
        if template == 0:
            tokens = [person, "lives", "in", city]
            tags = ["B-PER", "O", "O", "B-LOC"]
        elif template == 1:
            tokens = [person, rng.choice(_SURNAMES), "visited", *multi]
            tags = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]
        elif template == 2:
            tokens = [person, "came", "from", city]
            tags = ["B-PER", "O", "O", "B-LOC"]
        elif template == 3:
            tokens = ["the", "city", "of", city, "is", "large"]
            tags = ["O", "O", "O", "B-LOC", "O", "O"]
        elif template == 4:
            tokens = [person, rng.choice(_SURNAMES), "stayed", "in", city]
            tags = ["B-PER", "I-PER", "O", "O", "B-LOC"]
        else:
            tokens = [person, "flew", "to", *multi]
            tags = ["B-PER", "O", "O", "B-LOC", "I-LOC"]
        out.append((tokens, tags))
    return out


@dataclass(frozen=True)
class ScaleRow:
    """One measured workload: size, timing, throughput, memory, and quality."""

    size: int
    seconds: float
    throughput: float
    peak_mib: float
    f1: float


def _train() -> CRF:
    return Trainer(LABELS, TrainingConfig(epochs=25, seed=0)).train(train_data())


def _peak_mib(crf: CRF, size: int) -> float:
    """Peak memory to hold ``size`` sentences and run a single Viterbi decode."""

    tracemalloc.start()
    tracemalloc.reset_peak()
    data = _generate(size, seed=0)
    crf.viterbi(data[0][0])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / _BYTES_PER_MIB


def run_scale(sizes: tuple[int, ...] = WORKLOAD_SIZES) -> list[ScaleRow]:
    """Train once, then measure Viterbi decoding scaling across sizes."""

    crf = _train()
    rows: list[ScaleRow] = []
    for size in sizes:
        data = _generate(size, seed=0)
        start = time.perf_counter()
        report = evaluate_spans(crf, data)
        elapsed = time.perf_counter() - start
        rows.append(
            ScaleRow(
                size=size,
                seconds=elapsed,
                throughput=size / elapsed if elapsed else 0.0,
                peak_mib=_peak_mib(crf, size),
                f1=report.f1,
            )
        )
    return rows


def main() -> None:
    print("Synthetic templated BIO sentences; from-scratch linear-chain CRF.")
    print("Pure-Python Viterbi; not benchmarked against compiled sequence libraries.")
    print(f"{'sentences':>10}  {'seconds':>9}  {'sent/sec':>9}  {'peak MiB':>9}  {'span F1':>9}")
    for row in run_scale():
        print(
            f"{row.size:>10d}  {row.seconds:>9.3f}  {row.throughput:>9.1f}  "
            f"{row.peak_mib:>9.2f}  {row.f1:>9.3f}"
        )


if __name__ == "__main__":
    main()
