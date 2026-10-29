"""Observation feature extraction for the linear-chain CRF.

Each token position produces a set of string feature keys describing the token
and its local context (the word, its shape, prefixes and suffixes, and the
previous and next words). The CRF learns a weight per (feature, label) pair for
these emission features and a weight per (previous label, label) pair for
transitions. Features are deliberately simple and transparent.
"""

from __future__ import annotations

from collections.abc import Sequence


def _shape(token: str) -> str:
    shape = []
    for character in token:
        if character.isupper():
            shape.append("X")
        elif character.islower():
            shape.append("x")
        elif character.isdigit():
            shape.append("d")
        else:
            shape.append("-")
    return "".join(shape)


def token_features(tokens: Sequence[str], position: int) -> list[str]:
    """Return the emission feature keys for the token at ``position``."""
    token = tokens[position]
    lower = token.lower()
    features = [
        "bias",
        f"word={lower}",
        f"shape={_shape(token)}",
        f"prefix2={lower[:2]}",
        f"suffix2={lower[-2:]}",
        f"suffix3={lower[-3:]}",
        f"is_title={token[:1].isupper()}",
        f"is_digit={token.isdigit()}",
    ]
    if position > 0:
        features.append(f"prev_word={tokens[position - 1].lower()}")
    else:
        features.append("BOS")
    if position < len(tokens) - 1:
        features.append(f"next_word={tokens[position + 1].lower()}")
    else:
        features.append("EOS")
    return features


def sentence_features(tokens: Sequence[str]) -> list[list[str]]:
    """Return the emission features for every position in a sentence."""
    return [token_features(tokens, position) for position in range(len(tokens))]
