# Operations Guide - CRF Sequence Labeler

The operational contract of the tagging service.

## Running the service

```bash
uv run python -m crf_sequence_labeler serve --host 0.0.0.0 --port 8000
```

A standard-library HTTP server with **zero runtime dependencies**. The CRF is trained once at
startup and then serves stateless tagging requests.

## Endpoints

| Method | Path | Body | Success |
|---|---|---|---|
| GET | `/health` | none | status and label set (liveness) |
| GET | `/ready` | none | readiness once the CRF is trained |
| GET | `/metrics` | none | per-route request, error, and latency metrics (Prometheus text) |
| POST | `/tag` | `{"tokens": [str]}` | BIO tags and extracted entity spans |
| GET | `/evaluate` | none | span-level precision, recall, and F1 on held-out data |

Errors return JSON `{"error": ...}` with a 4xx status; malformed or overlong token lists are
rejected.

## Service level objectives

Measured by `benchmarks/stress.py` on a developer laptop:

| Objective | Target | Observed (single worker) |
|---|---|---|
| Tag latency p50 | < 1 ms | about 0.03 ms |
| Tag latency p99 | < 1 ms | about 0.07 ms |
| Throughput | > 10000 tags/second/core | about 25000/second |
| Startup training | < 1 s | about 40 ms |

## Operational constraints

- **Training is a startup cost; decoding is cheap.** The CRF trains in tens of milliseconds;
  each tag is a Viterbi pass whose cost is linear in sentence length and quadratic in the
  (small) label count.
- **The decode path is CPU bound and holds the global interpreter lock.** Throughput is flat
  across threads; scale with processes.
- **Requests are bounded**: token lists are capped at 200 tokens.
- **No persistence, no authentication.** Front the service with a gateway.

## Failure modes and responses

| Symptom | Likely cause | Response |
|---|---|---|
| Poor tags on new domain | Out-of-domain text | Retrain on in-domain data; add features |
| `400` on tag | Empty, non-string, or overlong tokens | Inspect the `error` field |
| Rising p99 under load | Concurrency above one worker per core | Add replicas; cap concurrency |

## Capacity planning

A single core serves tens of thousands of tags per second. Scale with process-per-core
replicas; the service is stateless.
