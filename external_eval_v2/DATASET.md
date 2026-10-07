# Independent External Evaluation Dataset v2

- Raw candidates: 300
- Visually accepted evaluation images: 220
- Everyday-life strata: 20
- Maximum downloaded width: 1280 px

The frozen evaluation list is `manifest.csv`. `manifest_candidates.csv` retains
all 300 raw candidates for auditability; rejected downloads remain in `images/`
but are not part of evaluation.

The accepted set covers roads and transit, signs, homes, kitchens, shops, food,
offices and schools, screens and controls, charts and maps, workshops, clothing,
people, sport and play, plants, landscapes, animals, architecture, night scenes,
weather, and small coloured objects.

This set is evaluation-only. Do not use it for training, checkpoint selection,
threshold tuning, or post-hoc sample replacement. It excludes items from the
original 60-image set and records source, license, author, dimensions, SHA-1,
and SHA-256. See `REVIEW.md` for the frozen visual-review exclusions.

Run `python verify_dataset.py` before scoring.
