# System Card - CRF Sequence Labeler

## Overview

- **What it is:** a linear-chain conditional random field for sequence labeling, trained from
  scratch, with Viterbi decoding and span-level evaluation.
- **What it is not:** a neural tagger; features are hand-designed and transparent.
- **Version:** 0.1.0.

## Intended use

- **Primary use cases:** learning how CRFs are trained and decoded; tagging short sequences
  with named entities or parts of speech on modest datasets.
- **Primary users:** engineers and students studying structured prediction.

## Out-of-scope / prohibited use

- Not for web-scale corpora (pure-Python training).
- Accuracy depends on the hand-designed features and the training data.

## Performance

See the README: span-level F1 of about 0.857 on held-out named-entity sentences.

## Risks and limitations

- Hand-designed features cap accuracy versus neural extractors.
- Small, synthetic training data limits coverage.

## Safety mitigations

- Deterministic under a fixed seed; offline; bounded token lists in the service.

## Environmental / cost footprint

Negligible; training runs in tens of milliseconds on the bundled data.
