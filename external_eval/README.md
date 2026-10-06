# External evaluation set

This directory contains the reproducible collection metadata for the CVDLens
out-of-domain evaluation set. The image files are intentionally excluded from
Git; run the collector to recreate the local set.

```powershell
powershell -ExecutionPolicy Bypass -File external_eval/collect_commons.ps1
```

The collector downloads 10 images for each of six categories from Wikimedia
Commons at up to 1280 pixels wide. It accepts JPEG, PNG, and WebP files with a
minimum original resolution of 640x480 and records the author, source page,
license, URLs, dimensions, and SHA-256 hash in `manifest.csv`.

The set is evaluation-only. Do not use these images for training, checkpoint
selection, threshold tuning, or post-hoc replacement based on model results.

## Run deployment-equivalent validation

```powershell
.\.eval-env\python.exe external_eval\evaluate.py
```

The evaluator calls the deployed P/D ONNX models and the deployed T hue rule,
then writes per-image measurements and an aggregate report under `results/`.
CRR values are omitted for images whose confusion-weight map is empty.

Before publishing a thesis or redistributing an image, manually review its
Wikimedia Commons file page and comply with its current license and attribution
requirements. The API metadata is an aid, not legal advice.
