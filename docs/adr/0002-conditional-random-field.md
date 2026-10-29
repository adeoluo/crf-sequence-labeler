# ADR-0002: A conditional random field over independent classifiers

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

Sequence labeling can be done with a per-token classifier or with a model that scores the
whole label sequence jointly. Per-token classifiers ignore label-to-label structure.

## Decision

Use a linear-chain conditional random field: emission weights per (feature, label) and
transition weights per (previous label, label), with the sequence probability normalized by
the partition function. Decode with Viterbi; train on the conditional log-likelihood.

## Alternatives considered

- **Per-token maximum entropy / logistic regression.** Simple, but cannot learn that, for
  example, an I-LOC tag must follow a B-LOC, producing invalid label sequences.
- **A hidden Markov model.** Generative and limited to simple emission features; a CRF is
  discriminative and admits rich overlapping features.

## Consequences

- The model learns transition structure and produces valid label sequences.
- Inference needs dynamic programming (Viterbi and forward-backward), implemented here.
- Training is more involved than a per-token classifier, which the next ADR addresses.
