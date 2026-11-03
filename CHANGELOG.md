# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.1.0] 2026-11-03

### Features

- Add CRF features, Viterbi, forward-backward, and gradient training
- Add a named-entity dataset, span-level evaluation, and CLI
- Add HTTP service with observability, benchmarks, and documentation

### Bug fixes

- Skip empty sentences in training and make seed meaningful (#10)

### Documentation

- Use plain, factual wording (#9)
- Standard README structure and language framing (#12)
- Vertical layered architecture diagram
- Horizontal layered architecture diagram
- Compact horizontal architecture diagram
- Recorded demos of more commands, with their vhs tapes (#1)
- Recorded demos of the HTTP service and more commands (#2)
- Describe the supply chain accurately (#3)

### Tests

- Alias imported data loaders so pytest does not collect them

### Build and CI

- Run CodeQL on main and on a weekly schedule only

### Maintenance

- Scaffold crf-sequence-labeler project
- Bump continuous integration action versions and dev dependencies
- Remove Dependabot configuration
- Release tooling, changelog and agent guides (#4)
