"""Tests for span extraction, span-level evaluation, and the CLI."""

from __future__ import annotations

from crf_sequence_labeler import LABELS, evaluate_spans, extract_spans
from crf_sequence_labeler import test_data as held_out_data
from crf_sequence_labeler import train_data as training_data
from crf_sequence_labeler.cli import main
from crf_sequence_labeler.training import Trainer, TrainingConfig


def test_extract_spans_bio() -> None:
    tags = ["B-PER", "I-PER", "O", "B-LOC"]
    assert extract_spans(tags) == {("PER", 0, 2), ("LOC", 3, 4)}
    assert extract_spans(["O", "O"]) == set()


def test_extract_spans_handles_boundary_entity() -> None:
    assert extract_spans(["O", "B-LOC", "I-LOC"]) == {("LOC", 1, 3)}


def test_crf_generalizes_on_held_out_spans() -> None:
    crf = Trainer(LABELS, TrainingConfig(epochs=25, seed=0)).train(training_data())
    report = evaluate_spans(crf, held_out_data())
    assert report.f1 > 0.6  # generalizes to unseen names and places
    assert report.precision > 0.0 and report.recall > 0.0


def test_crf_tags_unseen_sentence() -> None:
    crf = Trainer(LABELS, TrainingConfig(epochs=25, seed=0)).train(training_data())
    tags = crf.viterbi(["Oscar", "lives", "in", "Vienna"])
    assert tags[0] == "B-PER"
    assert tags[3] == "B-LOC"


def test_cli_evaluate(capsys) -> None:
    main(["--epochs", "15", "evaluate"])
    out = capsys.readouterr().out
    assert "F1" in out


def test_cli_tag(capsys) -> None:
    main(["--epochs", "15", "tag", "Alice", "lives", "in", "Paris"])
    out = capsys.readouterr().out
    assert "B-PER" in out
    assert "B-LOC" in out
