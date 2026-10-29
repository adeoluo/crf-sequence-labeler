"""A production-style stress benchmark for the tagging service.

Trains once, then drives the tag handler under sustained, concurrent load and
reports throughput with p50, p95, and p99 latency. Deterministic and offline;
run with:

    uv run python benchmarks/stress.py
"""

from __future__ import annotations

import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from crf_sequence_labeler.service import TaggerService, handle

_SENTENCES = (
    ["Alice", "lives", "in", "Paris"],
    ["Bob", "Smith", "visited", "Rome"],
    ["Carol", "flew", "to", "Tokyo"],
    ["the", "city", "of", "Berlin"],
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    service = TaggerService.trained(epochs=25)
    rng = random.Random(0)
    requests = [{"tokens": list(rng.choice(_SENTENCES))} for _ in range(3000)]

    def one(body: dict[str, object]) -> float:
        start = time.perf_counter()
        handle("POST", "/tag", body, service)
        return (time.perf_counter() - start) * 1000

    print(f"Requests: {len(requests)}")
    for workers in (1, 4, 8):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            latencies = list(pool.map(one, requests))
        wall = time.perf_counter() - start
        print(
            f"workers={workers:<2}  "
            f"throughput={len(requests) / wall:>7.0f} req/s  "
            f"p50={statistics.median(latencies):>6.3f} ms  "
            f"p95={_percentile(latencies, 0.95):>6.3f} ms  "
            f"p99={_percentile(latencies, 0.99):>6.3f} ms"
        )


if __name__ == "__main__":
    main()
