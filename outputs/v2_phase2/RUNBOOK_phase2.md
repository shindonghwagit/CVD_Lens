# Phase 2 RUNBOOK — Browser Deployment

Goal: ship the Phase 1 model (`model_best.pt`, step 9000) as a real-time,
in-browser CVD correction filter, then evaluate it against Daltonize.

Deploy stack (from project memory): Next.js + ONNX Runtime Web (WebGL/WASM) +
Tailwind, Vercel. Model path: PyTorch → ONNX → (later) fp16.

---

## Step 1 — ONNX export + numeric parity  ✅ DONE

Self-contained per-type graphs; browser feeds only `(srgb, severity)`.

**Artifacts** (`outputs/v2_phase2/`):
- `cvdlens_p.onnx` / `cvdlens_d.onnx` / `cvdlens_t.onnx` — 847 KB each
- `parity_report.json` — parity + latency data

**Completion criteria & results:**
| criterion | target | result |
|---|---|---|
| one static graph per type (P/D/T) | 3 files | ✅ 3 |
| input shape | static (1,3,256,256), opset 16, no dynamic_axes | ✅ |
| severity a live input (not frozen) | 0.5 vs 1.0 output differs | ✅ Δout 0.20 / 0.22 / 0.56 |
| file size | < ~1 MB each | ✅ 847 KB |
| PyTorch ↔ ORT parity | max\|diff\| < 1e-3 | ✅ **2.04e-05** |
| ORT CPU latency (baseline) | measured | ✅ median 21–40 ms |

**Design notes:**
- `model.wrap_for_onnx(model, cvd_type)` freezes the type and computes the
  confusion weight `w` *inside* the graph (Lab ΔE of original vs Machado-sim,
  thresholded + blurred). Severity drives both FiLM and the Machado matrix via
  `simulation.machado_matrix_tensor` (tent-weight interpolation — ONNX-traceable,
  exact match to the float path at keypoints; the 2e-5 parity residual is
  fp-accumulation from the two Machado code paths, not a logic gap).
- Reproduce: `py -m cvdlens_v2.export_onnx` then `py -m cvdlens_v2.parity_check`.

**Latency baseline (ORT CPU, 256×256, median of 20):** T 21 ms · D 34 ms · P 40 ms.
This is the CPU reference for the real-time target; browser WASM/WebGL will
differ (WebGL likely faster on GPU, WASM comparable to slower than CPU ORT).
Decision point for Step 2: if WASM misses the frame budget, add fp16 quantization
and/or downscale the inference resolution.

---

## Step 2 — inference integration  ✅ DONE (SERVER inference retained)

**Outcome: correction runs on the server (FastAPI on Render), now serving the new
Phase 1 per-type models.** A full in-browser (ort-web) integration was built and
then reverted; the app keeps its original server round-trip.

### Decision: browser (6a6fc3c) → server (75fac88)

The ort-web browser integration was implemented and verified end-to-end
(`tsc` 0 errors, `next build` ✓, ort-web↔PyTorch parity **1.0e-6**, severity live)
in commit `6a6fc3c`. It was then reverted (`75fac88`) and the backend updated to the
new models (`659e643`). **Reason for keeping server inference: the FastAPI backend was
already deployed and running on Render** — reusing the existing, working deployment
path was preferred over shipping a new browser-inference stack for this milestone.
No browser-vs-server latency measurement drove this; it was an infrastructure-reuse
decision. The browser-inference work is **not discarded**: it is preserved in commit
`6a6fc3c` and under `outputs/v2_phase2/` (3 exported ONNX graphs, parity report,
`cvdEngine.ts`), so a later switch to browser-only or a **hybrid** (browser default +
server fallback) can reuse it directly.

**Exam Q&A draft — "왜 서버 추론인가?"**
> 브라우저 추론(ONNX Runtime Web) 경로를 구현·검증까지 마쳤으나(PyTorch 대비 수치 오차
> 1e-6, severity 실시간 반영 확인), 최종 배포는 **이미 Render에 구축·운영 중이던 FastAPI
> 백엔드를 재사용**하는 서버 추론으로 정했다. 서버 방식의 단점은 인정한다 — Render 무료
> 티어의 콜드스타트(idle 후 첫 요청 ~50초)와 서버 왕복 지연. 대신 배포 파이프라인이 이미
> 검증돼 있어 안정적이고, 기기별 WebGPU/WASM 편차나 첫 로드(wasm ~13.5MB) 부담이 없다.
> 브라우저 경로는 언제든 전환 가능한 검증 완료 자산으로 보존해 두었으므로, 서버 비용·지연이
> 문제가 되면 브라우저 또는 하이브리드로 이행하면 된다.

### What actually shipped (server path)
- `cvd-lens/inference/main.py` — loads `cvdlens_{p,d,t}.onnx` (step-9000); `/infer` +
  `/infer/video` route by `cvd_type`; `severity` form field (default 1.0). Old 4-channel
  `cvdlens_fp32.onnx` (26.5 MB) removed. Verified locally: `/health` ok, `/infer` returns
  a valid 256×256 RGB JPEG for p/d/t, severity changes output.
- Frontend: reverted to the original server contract (`infer(imageData, cvd_type)` →
  `POST /infer` → JPEG). Deploy: `git push` → Render rebuilds backend (new models),
  Vercel redeploys frontend (behaviour unchanged). Vercel `NEXT_PUBLIC_API_URL` already
  points at Render — no env change needed.

### Step 2 add-ons (frontend, inference-location-independent)
- **severity slider** (ImageCorrection) → server `severity` form field (300 ms debounce).
- **CVD-sim compare view** — `lib/cvdSim.ts` (pure Brettel, no ort-web import): toggle shows
  sim(original) vs sim(corrected) side by side; computed client-side (no server round-trip).

### Step 2.5 — real-time camera stream  ⏳ NOT STARTED (out of scope, await instruction)

## Step 3 — Quantitative evaluation + Daltonize comparison  ✅ DONE

CRR-NP head-to-head vs `simulation.daltonize` (Brettel/error-shift, the single
in-repo target generator; no external daltonize libs). 60 held-out COCO images
(top/mid/low confusion-mass, seed 20260803), disjoint from the Phase 1 bank (10)
and held-out set (8). 60 × 3 types × 2 methods = 360 cases.

**Code** (`cvdlens_v2/`): `step3_eval_set.py` (stratified sampler),
`step3_metrics.py` (CRR=`ratio_w`, NP=|Δ|+LPIPS-vgg, secondary SI_uniform/
corr_guide), `step3_eval.py` (runner + scatter + witnesses + report).
**Artifacts** (`outputs/v2_phase3/`): `eval_set.json`, `eval_results.json`,
`step3_scatter.png`, `step3_case_{bigwin,similar,lose}.png`, `step3_report.md`.

Fairness: both methods at 256², same simulator (machado, sev 1.0), same
confusion weight w (function of the original only).

**Result — hypothesis SUPPORTED (reported as measured):**
| axis | CVDLens | Daltonize | note |
|---|---|---|---|
| CRR (top+mid, w-meaningful) | **1.353** | 1.029 | daltonize ≈ no net weighted-contrast gain on the CVD view |
| NP \|Δ\| (all tiers) | **0.0192** | 0.0308 | less damage; Wilcoxon p<0.05 every type |
| NP LPIPS | **0.0458** | 0.0664 | perceptually closer to original |
| low-tier \|Δ\| (do-nothing) | **0.0010** | 0.0043 | CVDLens leaves low-confusion images ~untouched |

CVDLens wins BOTH axes: recovers more contrast where it matters AND damages
less. This confirms the Phase 0 preview ("daltonize destroys P/D local contrast,
6.23→4.34"). Negative-honesty: one w-meaningful case (`000000183104`/p) has
CVDLens recovering *slightly* less than daltonize (ΔCRR −0.026) — but even there
CVDLens is more natural (|Δ| −0.019, LPIPS −0.035). See step3_report.md.
Caveat: low-tier `ratio_w` is numerically unstable (w̄≈0) → used only as a
naturalness check, excluded from the CRR verdict. This is the paper's
comparison-section data.

Next (optional): Step 2.5 real-time camera stream; paper write-up.

---

## Status
- Step 1: **complete** (export + parity + baseline bench).
- Step 2: **complete** (server inference retained; ONNX browser path preserved).
- Step 3: **complete** (CRR/NP vs daltonize — hypothesis supported).
- Next: Step 2.5 (camera stream) / paper write-up on instruction.
