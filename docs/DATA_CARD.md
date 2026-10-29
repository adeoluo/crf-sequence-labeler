# Data Card - CRF Sequence Labeler

The project bundles one small, synthetic named-entity dataset. There is no external data and
nothing is downloaded.

## bundled dataset (`src/crf_sequence_labeler/data.py`)

- **Provenance:** authored for this project. Sentences tagged in the BIO scheme over person
  (PER) and location (LOC) entities, with disjoint train and test splits.
- **Why this shape:** capitalization, position, and cue words (``in``, ``from``) make the
  entities learnable, and held-out names and places test generalization.
- **Licensing:** authored by this repository; MIT.
- **Known gaps:** small, English, two entity types; a demonstration dataset.

## Ethical considerations

The dataset is synthetic with fictitious names and public place names; no personal or
sensitive data.
