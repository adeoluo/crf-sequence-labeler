"""A bundled named-entity dataset with BIO tags for demonstration and evaluation.

Sentences are tagged in the BIO scheme over two entity types, PER (person) and
LOC (location), plus O (outside). The data is synthetic but structured so that
capitalization, position, and neighboring cue words (``in``, ``from``, ``said``)
make the entities learnable. Train and test splits are disjoint sentences.
"""

from __future__ import annotations

LABELS: tuple[str, ...] = ("O", "B-PER", "I-PER", "B-LOC", "I-LOC")

_TRAIN: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("Alice", "lives", "in", "Paris"), ("B-PER", "O", "O", "B-LOC")),
    (("Bob", "traveled", "to", "London"), ("B-PER", "O", "O", "B-LOC")),
    (("Carol", "Smith", "visited", "New", "York"), ("B-PER", "I-PER", "O", "B-LOC", "I-LOC")),
    (("David", "came", "from", "Berlin"), ("B-PER", "O", "O", "B-LOC")),
    (("Emma", "Jones", "stayed", "in", "Rome"), ("B-PER", "I-PER", "O", "O", "B-LOC")),
    (("Frank", "works", "in", "San", "Diego"), ("B-PER", "O", "O", "B-LOC", "I-LOC")),
    (("Grace", "flew", "to", "Tokyo"), ("B-PER", "O", "O", "B-LOC")),
    (("Henry", "Adams", "lives", "in", "Madrid"), ("B-PER", "I-PER", "O", "O", "B-LOC")),
    (("the", "city", "of", "Chicago", "is", "large"), ("O", "O", "O", "B-LOC", "O", "O")),
    (("Ivan", "said", "hello"), ("B-PER", "O", "O")),
)

_TEST: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("Julia", "lives", "in", "Vienna"), ("B-PER", "O", "O", "B-LOC")),
    (("Kevin", "Brown", "visited", "Oslo"), ("B-PER", "I-PER", "O", "B-LOC")),
    (("Laura", "traveled", "to", "Los", "Angeles"), ("B-PER", "O", "O", "B-LOC", "I-LOC")),
    (("Mike", "came", "from", "Dublin"), ("B-PER", "O", "O", "B-LOC")),
    (("Nina", "Clark", "stayed", "in", "Lisbon"), ("B-PER", "I-PER", "O", "O", "B-LOC")),
)


def _as_lists(
    rows: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
) -> list[tuple[list[str], list[str]]]:
    return [(list(tokens), list(tags)) for tokens, tags in rows]


def train_data() -> list[tuple[list[str], list[str]]]:
    """Return the labeled training sentences."""
    return _as_lists(_TRAIN)


def test_data() -> list[tuple[list[str], list[str]]]:
    """Return the held-out labeled test sentences."""
    return _as_lists(_TEST)
