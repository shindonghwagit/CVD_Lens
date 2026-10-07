# P/D external validation v2

Frozen independent set: 220 accepted images across 20 everyday-life strata.

| Type | Simulator | Relevant n | CRR mean (95% CI) | median | failures | success | NP MAE | large change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P | machado | 110 | 1.204 (1.174-1.239) | 1.166 | 0 | 100.0% | 0.0137 | 6.3% |
| P | brettel | 110 | 1.197 (1.166-1.233) | 1.156 | 0 | 100.0% | 0.0137 | 6.3% |
| D | machado | 110 | 1.167 (1.143-1.193) | 1.140 | 2 | 98.2% | 0.0159 | 6.9% |
| D | brettel | 110 | 1.155 (1.130-1.182) | 1.119 | 1 | 99.1% | 0.0159 | 6.9% |

CRR > 1 indicates increased local contrast in simulator-defined confusion regions. Results are engineering evidence, not a clinical user study.
