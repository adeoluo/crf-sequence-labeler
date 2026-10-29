# Evaluation Report - CRF Sequence Labeler

> Reproduce with `python -m crf_sequence_labeler evaluate`. Deterministic and offline.

## Setup

- **Data:** a BIO named-entity dataset (PER, LOC) with disjoint train and test splits.
- **Metric:** span-level precision, recall, and F1 - a predicted entity is correct only if
  its type and exact boundaries match a gold entity.
- **Model:** the linear-chain CRF trained for 25 epochs by log-likelihood gradient ascent.

## Results

| Metric | Value |
|---|---:|
| span-level F1 | 0.857 |
| precision | 0.818 |
| recall | 0.900 |

## Interpretation

The CRF reaches 0.857 span F1 on held-out sentences containing names and places it never saw
in training, correctly tagging them from shape, affix, and context features. Span-level
scoring is strict (exact type and boundaries), so this reflects genuine structured
prediction, not per-token accuracy inflation. The result confirms that the from-scratch
forward-backward gradient training learns the transition and emission structure of the task.

## Reproducibility

Training is seeded; the numbers regenerate exactly.
