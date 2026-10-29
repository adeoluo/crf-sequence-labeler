"""A dependency-free HTTP service exposing the trained CRF tagger.

The service trains the CRF once at startup over the bundled data and then tags
sentences and reports held-out evaluation. A pure handler is separated from the
socket binding for testing.

Endpoints
---------
``GET  /health``    liveness and label set.
``GET  /ready``     readiness once the CRF is trained.
``GET  /metrics``   per-route request, error, and latency metrics (Prometheus text).
``POST /tag``       body ``{"tokens": [str]}``; returns BIO tags and entity spans.
``GET  /evaluate``  span-level precision, recall, and F1 on held-out data.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from crf_sequence_labeler.crf import CRF
from crf_sequence_labeler.data import LABELS, test_data, train_data
from crf_sequence_labeler.evaluation import evaluate_spans, extract_spans
from crf_sequence_labeler.observability import ServiceMetrics
from crf_sequence_labeler.training import Trainer, TrainingConfig

JsonObject = dict[str, Any]
_MAX_TOKENS = 200


@dataclass(frozen=True)
class TaggerService:
    """Holds the trained CRF and answers tagging and evaluation requests."""

    crf: CRF
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def trained(cls, *, epochs: int = 25, seed: int = 0) -> TaggerService:
        crf = Trainer(LABELS, TrainingConfig(epochs=epochs, seed=seed)).train(train_data())
        return cls(crf=crf)

    def tag(self, tokens: list[str]) -> JsonObject:
        tags = self.crf.viterbi(tokens)
        spans = sorted(extract_spans(tags))
        return {
            "tokens": tokens,
            "tags": tags,
            "entities": [
                {
                    "type": entity_type,
                    "start": start,
                    "end": end,
                    "text": " ".join(tokens[start:end]),
                }
                for entity_type, start, end in spans
            ],
        }

    def evaluate(self) -> JsonObject:
        report = evaluate_spans(self.crf, test_data())
        return {
            "precision": round(report.precision, 6),
            "recall": round(report.recall, 6),
            "f1": round(report.f1, 6),
        }


def _route(
    method: str, path: str, body: Mapping[str, Any], service: TaggerService
) -> tuple[int, JsonObject]:
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", "labels": list(LABELS)}
    if method == "GET" and path == "/evaluate":
        return 200, service.evaluate()
    if method == "POST" and path == "/tag":
        tokens = body.get("tokens")
        if (
            not isinstance(tokens, list)
            or not tokens
            or not all(isinstance(t, str) for t in tokens)
        ):
            return 400, {"error": "'tokens' must be a non-empty list of strings"}
        if len(tokens) > _MAX_TOKENS:
            return 400, {"error": f"'tokens' exceeds {_MAX_TOKENS} tokens"}
        return 200, service.tag([str(t) for t in tokens])
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: TaggerService
) -> tuple[int, JsonObject]:
    """Route a request, recording per-route metrics, and return status and payload."""

    if method == "GET" and path == "/ready":
        return 200, {"status": "ready", "labels": list(LABELS)}
    start = time.perf_counter()
    status, payload = _route(method, path, body, service)
    service.metrics.observe(
        f"{method} {path}", (time.perf_counter() - start) * 1000.0, error=status >= 400
    )
    return status, payload


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: TaggerService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.service.metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": TaggerService.trained()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"crf-sequence-labeler service listening on http://{host}:{port}")
    server.serve_forever()
