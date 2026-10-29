# Threat Model - CRF Sequence Labeler

## Assets

- Integrity of the trained weights and the predicted tags and spans.

## Trust boundaries

- **Token lists are untrusted input to the service.** They are validated for type and length
  and only featurized and decoded, never executed.

## Threats and mitigations

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Unbounded computation | Very long token list | Resource use | Token count bounded to 200 |
| Malformed input | Non-string tokens | Crash | Type validation with a 400 |
| Irreproducible training | Unseeded gradient ascent | Flaky weights | Deterministic, seeded training |
| Supply-chain compromise | Malicious dependency | Code execution | Zero runtime dependencies; Dependabot and CodeQL in continuous integration |

## Residual risk

The tagger reflects its training data; on out-of-domain text accuracy drops. The bundled data
is synthetic and the limitation is a capability bound, not a security flaw.
