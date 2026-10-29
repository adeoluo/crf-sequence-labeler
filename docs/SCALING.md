# Scaling and Design Tradeoffs - CRF Sequence Labeler

How the reference design grows, and the tradeoffs taken.

## Reference architecture (today)

A single process trains a linear-chain CRF over an in-memory dataset and serves Viterbi
tagging over a standard-library HTTP service. Deterministic for a fixed seed.

## Scaling path

| Stage | Bottleneck | Change |
|---|---|---|
| One process | Decode CPU, global interpreter lock | One replica per core |
| Large label set | Quadratic forward-backward and Viterbi | Prune transitions; beam decoding |
| Large corpus | Pure-Python training | Train offline; load weights; use L-BFGS |
| Rich features | Feature count growth | Feature hashing; sparse updates |
| Many replicas | Repeated startup training | Train once; ship weights as an artifact |

## Substantial tradeoffs

- **CRF vs independent classifiers.** The CRF models transition structure and yields valid
  sequences, at the cost of dynamic-programming inference. Independent classifiers are cheaper
  but produce invalid tag sequences.
- **Full log-likelihood gradient vs structured perceptron.** The log-likelihood gradient uses
  forward-backward marginals and models probabilities, at a higher cost than the perceptron,
  which uses only Viterbi. The reference chooses the probabilistic, principled training.
- **Hand-designed vs neural features.** Hand features are transparent and dependency-free but
  cap accuracy; neural features generalize better at the cost of a heavy dependency.
- **Stochastic gradient ascent vs batch L-BFGS.** Stochastic updates are simple and online;
  L-BFGS converges in fewer passes but needs the full-batch gradient and more machinery.

## What would change under real load

Train offline with L-BFGS and ship the weights, serve from process-per-core replicas, and add
feature hashing and transition pruning for large label sets. The inference algorithms are
unchanged; only the training substrate and feature plumbing change.
