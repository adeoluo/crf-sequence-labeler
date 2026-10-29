"""Command line interface for the CRF sequence labeler."""

from __future__ import annotations

import argparse

from crf_sequence_labeler.data import LABELS, test_data, train_data
from crf_sequence_labeler.evaluation import evaluate_spans
from crf_sequence_labeler.training import Trainer, TrainingConfig


def _train(epochs: int, seed: int):  # type: ignore[no-untyped-def]
    return Trainer(LABELS, TrainingConfig(epochs=epochs, seed=seed)).train(train_data())


def _cmd_evaluate(args: argparse.Namespace) -> None:
    crf = _train(args.epochs, args.seed)
    report = evaluate_spans(crf, test_data())
    print("Span-level evaluation on held-out sentences:")
    print(f"  {report.format()}")


def _cmd_tag(args: argparse.Namespace) -> None:
    crf = _train(args.epochs, args.seed)
    tokens = args.tokens
    tags = crf.viterbi(tokens)
    print("Prediction:")
    for token, tag in zip(tokens, tags, strict=True):
        print(f"  {token:<14} {tag}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crf-sequence-labeler", description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("evaluate", help="span-level precision, recall, and F1 on held-out data")

    tag = sub.add_parser("tag", help="tag a sentence given as tokens")
    tag.add_argument("tokens", nargs="+", help="the whitespace-separated tokens to tag")

    handlers = {"evaluate": _cmd_evaluate, "tag": _cmd_tag}
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
