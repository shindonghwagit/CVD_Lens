# External validation report

Deployment-equivalent inference on 60 Wikimedia Commons images (10 per category).
P/D use the deployed ONNX models; T uses the deployed 30-degree hue rule. Images are aspect-preserving, capped at 512 px.

CRR > 1 means increased contrast in CVD-confusion regions. NP MAE is mean absolute sRGB change (lower is more natural).
The `meaningful half` is the 30 images at or above each type's median confusion-weight mass; this avoids unstable CRR on irrelevant images.

| Type | CRR all mean/median | CRR meaningful mean/median | failures (<1) | NP MAE | changed pixels | large changes |
|---|---:|---:|---:|---:|---:|---:|
| P | 1.148/1.113 | 1.224/1.182 | 1/30 | 0.0137 | 58.7% | 6.3% |
| D | 1.126/1.095 | 1.192/1.152 | 0/30 | 0.0156 | 59.8% | 6.4% |
| T | 1.032/1.009 | 1.007/0.997 | 16/30 | 0.0159 | 46.8% | 10.4% |

## Interpretation

- P and D generalized well on the confusion-relevant half: mean CRR was 1.224 and 1.192, with 1 and 0 images below 1 respectively.
- T did not generalize reliably: its relevant-half median was 0.997 and 16/30 images were below 1. The worst relevant cases were `traffic_02` (0.762), `nature_03` (0.781), and `indoor_lowlight_05` (0.801).
- T also made the largest local changes: pixels changing by more than 0.1 averaged 10.4%, versus 6.3-6.4% for P/D. `nature_03` changed 95.9% of pixels by more than 0.1 while CRR fell to 0.781.

Conclusion: the external set supports the deployed P/D models as an engineering result. It does not support claiming that the current T hue rule consistently improves arbitrary photographs; T should be presented as an experimental heuristic or revised and revalidated.

## Limitations

This is an engineering robustness check, not a clinical user study. The set is small and web-sourced; CRR is simulator-based and does not prove human perceptual benefit.
