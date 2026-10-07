# Deployed API E2E parity

- Endpoint: `https://cvd-lens.onrender.com/infer`
- Images: 220 (full frozen manifest)
- HTTP requests: 440 (P and D)
- Verdict: **PASS**

| Check | Result |
|---|---|
| http_200 | PASS |
| jpeg_response | PASS |
| dimensions_match | PASS |
| pixel_parity_mae_le_2_255 | PASS |
| pixel_parity_p99_le_4_255 | PASS |

- Mean pixel parity MAE: 0.000000
- Worst pixel parity MAE: 0.000001
- Worst pixel parity p99: 0.000000
- Mean latency: 4.358 s
- Maximum latency: 10.529 s
