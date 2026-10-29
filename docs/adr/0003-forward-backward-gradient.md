# ADR-0003: Forward-backward log-likelihood gradients, no autodiff

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

Training a CRF requires the gradient of the conditional log-likelihood, which involves the
expected feature counts under the model. Those expectations need marginal probabilities.

## Decision

Compute node and edge marginals with the forward-backward algorithm in log space, form the
gradient as empirical minus expected feature counts, and apply stochastic gradient ascent
with an L2 penalty. All derivatives are written out by hand.

## Alternatives considered

- **Automatic differentiation.** Convenient but a heavy dependency and opaque for a reference
  whose purpose is to show the CRF mechanics.
- **Structured perceptron.** Simpler (uses only Viterbi, no partition function) but does not
  model probabilities or use marginals; the full CRF gradient is the correct choice.

## Consequences

- The project is dependency-free and the training mathematics is fully visible.
- Forward-backward is cubic per sentence in the label count, which is fine for small tag sets.
- Log-space arithmetic with log-sum-exp keeps the computation numerically stable.
