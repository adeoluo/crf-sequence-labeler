# End-to-End Case Study - Tagging Entities a Model Never Saw

How a conditional random field labels names and places it was never trained on.

## Scenario

A team needs named-entity tagging but has only a small labeled set and no pretrained model.
The question is whether a from-scratch CRF can generalize beyond its training words.

## Tagging

```bash
curl -s -X POST localhost:8000/tag -d '{"tokens": ["Oscar", "lives", "in", "Vienna"]}'
```

```json
{"tokens": ["Oscar", "lives", "in", "Vienna"],
 "tags": ["B-PER", "O", "O", "B-LOC"],
 "entities": [{"type": "PER", "start": 0, "end": 1, "text": "Oscar"},
              {"type": "LOC", "start": 3, "end": 4, "text": "Vienna"}]}
```

Neither "Oscar" nor "Vienna" was in the training data. The CRF tags them correctly because its
features capture capitalization, affixes, and the cue word "in", and its transition weights
know a location follows such a cue.

## Measuring generalization

```bash
curl localhost:8000/evaluate
# {"precision": 0.818, "recall": 0.900, "f1": 0.857}
```

On a held-out set of unseen entities the CRF reaches 0.857 span F1 under strict span scoring
(exact type and boundaries), confirming it learned structure, not words.

## What an operator sees

- **A cheap, fast tagger.** Training is tens of milliseconds; tagging is tens of thousands of
  sentences per second.
- **Explicit entity spans.** The response returns typed spans with offsets and text, ready to
  consume downstream.
- **An honest metric.** Span-level F1 is strict and reported on held-out data.

## Takeaway

A linear-chain CRF, trained from the gradient up, generalizes to unseen entities through
feature structure, which is exactly why conditional random fields were the workhorse of
sequence labeling before neural models.
