"""Tests for the dependency-free HTTP tagging service (pure handler layer)."""

from __future__ import annotations

from crf_sequence_labeler.service import TaggerService, handle

_SERVICE = TaggerService.trained(epochs=20)


def test_health_and_ready() -> None:
    status, payload = handle("GET", "/health", {}, _SERVICE)
    assert status == 200 and "B-PER" in payload["labels"]
    status, payload = handle("GET", "/ready", {}, _SERVICE)
    assert status == 200 and payload["status"] == "ready"


def test_tag_returns_entities() -> None:
    status, payload = handle(
        "POST", "/tag", {"tokens": ["Alice", "lives", "in", "Paris"]}, _SERVICE
    )
    assert status == 200
    types = {entity["type"] for entity in payload["entities"]}
    assert "PER" in types and "LOC" in types


def test_evaluate() -> None:
    status, payload = handle("GET", "/evaluate", {}, _SERVICE)
    assert status == 200
    assert 0.0 <= payload["f1"] <= 1.0


def test_tag_rejects_bad_tokens() -> None:
    status, _ = handle("POST", "/tag", {"tokens": []}, _SERVICE)
    assert status == 400
    status, _ = handle("POST", "/tag", {"tokens": [1, 2]}, _SERVICE)
    assert status == 400


def test_metrics_and_unknown_route() -> None:
    service = TaggerService.trained(epochs=15)
    handle("POST", "/tag", {"tokens": ["Bob", "runs"]}, service)
    text = service.metrics.render_prometheus()
    assert "service_requests_total" in text
    assert 'route="POST /tag"' in text
    status, payload = handle("GET", "/nope", {}, service)
    assert status == 404 and "error" in payload
