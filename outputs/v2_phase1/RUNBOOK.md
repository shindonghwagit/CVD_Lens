# Phase 1 — Training Runbook

Preflight status (this must be current before you launch a full run):

| item                                    | status  |
|-----------------------------------------|---------|
| Loss framework (Phase 0 v2)             | CLOSED  |
| Model unit tests (a, b, c)              | PASS    |
| Forward/backward smoke                  | PASS    |
| ONNX static export (opset 16, gather-only slice) | PASS |
| ORT parity (PyTorch vs onnxruntime)     | max\|diff\| ≈ 1.6e-07 |
| Kaggle wrapper (`cvdlens_v2/kaggle_train.py`) | READY |
| `train.py --resume` (12h-safe)          | READY   |

## Launch order

Do **NOT** kick off a full 20k-iter run cold — the val curve direction has
never been observed on this architecture. First a short preflight, then
scale up.

### Step 1: Preflight — `iters=2000`

**Purpose**: confirm the val curves are moving in the right direction
before spending 12+ GPU-hours.

Kaggle T4×2 (recommended):

```
!cd /kaggle/working/graduation_project && \\
    python -m cvdlens_v2.kaggle_train --iters 2000 --val-every 500 \\
        --log-every 50 --batch-size 16 --pretrained
```

Local single GPU (if available):

```
py -m cvdlens_v2.train \\
    --data-dir <COCO_TRAIN_DIR> \\
    --out-dir outputs/v2_phase1_preflight \\
    --iters 2000 --val-every 500 --log-every 50 \\
    --batch-size 16 --lr 2e-4 --lambda-tv 0.03 --pretrained
```

**Expected val trajectory** (checked at 500 / 1000 / 1500 / 2000):

| metric        | direction    | red flag                        |
|---------------|--------------|---------------------------------|
| ratio_w (P)   | ↑ from 1.000 | still 1.00 at step 1500         |
| ratio_w (D)   | ↑ from 1.000 | ↑ but slower than P             |
| ratio_w (T)   | ↑ from 1.000 | ↑ (should be fastest)           |
| SI            | ≤ 0.15 all steps | > 0.20 at any step          |
| 1761 \|Δ\|    | < 0.005 all steps | > 0.01 at any step         |
| L_c curve     | ↓            | flat or ↑ = optimizer not moving |

If any red flag hits → STOP. Debug before scaling. Do not proceed to
step 2.

### Step 1.5: Resume round-trip verification

The first Full-run session doubles as a **resume round-trip test**. Even
if training then runs cleanly for hours, a silent resume failure means
the second session starts from scratch and burns 12h. Verify **before
walking away** from the machine on the second commit:

- **Check A** — Notebook cell 2 log must show:
      `[resume] checkpoint step=2000`
  (or whatever step the preflight ended at). If it prints
  `[fresh] no prior checkpoint attached`, the previous version's Notebook
  Output was not added to Input. Cancel and re-attach.
- **Check B** — Cell 4 (`tr.main(args)`) first log line must be at
  **step 2000+**, not `step 1`. If it prints `[step 1/20000]`, the
  in-process resume path in `train.py` failed — cancel immediately,
  do NOT let a 12h session grind through a duplicated preflight range.
- **Check C₀** — Immediately after resume, cell emits:
      `[schedule] step=2000/20000  lr=1.9X-e04  T_max=20000  eta_min=2.0e-05`
  - `lr` must be close to `args.lr` (2e-4), lightly cosine-decayed to
    ~1.8-1.9e-4 for step 2000 of 20000. If it prints `lr=2.00e-05`,
    the `CosineAnnealingLR` T_max persistence bug hit — training will
    crawl at eta_min for the rest of the run. Cancel and check that
    the T_max override code (`scheduler.T_max = total_steps`) is in
    place.
  - `T_max` must equal current `--iters` (20000), NOT the preflight's
    2000.
- **Check C₁** — First validation block (step 3000 for ITERS=20000)
  `ratio_w` values must **continue smoothly** from where preflight
  left off (e.g., P was 1.05 at end of preflight → step 3000 shows
  1.06-1.08, not jumping back to 1.02). A discontinuous drop suggests
  the optimizer state was not restored (only weights were), and the
  Adam momentum history is wrong. Report immediately — the checkpoint
  schema in `train.py` includes `optim` and `scheduler` keys; if
  those are missing, resume is degraded and the run should be
  aborted.

### Val output format (post-Phase-1 SI diagnostic revision)

Each row: `[mark]*stem type |Δ| rw SIu (SI SI_abs corr)`

- `mark`: `P` or `F` (hard gate). Trailing `W` = SI_uniform soft
  warning. `*` after `]` = LOW-W do-nothing anchor (1761).
- **Hard-gate criteria (block PASS)**: ratio_w, identity guard,
  do-nothing.
- **Soft criterion (warning only)**: SI_uniform < 0.35. Computed
  only in truly-uniform-original regions (w>0.3 AND |∇luma|<0.02);
  full SI on natural images is dominated by guide-aligned edge
  response and is unusable as a hard gate (see phase1 SI diagnostic).
- **Diagnostic-only** columns: full-image SI, SI_abs (numerator),
  corr_guide (0.5+ = high-freq tracks luma edges, expected).

Only after A, B, C all pass, let the session run unattended.

### Step 2: Full run — `iters=20000`

Only after step 1 shows healthy curves:

```
!cd /kaggle/working/graduation_project && \\
    python -m cvdlens_v2.kaggle_train --iters 20000 --resume
```

`--resume` picks up from `model_latest.pt` — safe across Kaggle's 12h
session boundary. Re-run the exact same command in a fresh session; it
continues from the last saved step.

## Phase 1 success criteria (baked into `train.py::validate`)

Per CVD-active image (9 of 10):
- ratio_w  P ≥ 1.10   D ≥ 1.13   T ≥ 1.27
    (recalibrated from Exp 1-C tv=0.1 row − 0.03; network is
     structurally incapable of matching the raw-field tv=0.03 numbers
     because those included speckle inflation)
- SI < 0.15   (network target, well below raw-field 0.26–0.34)
- \|Δ\| > 0.01

Low-w (image 1761, w̄=0.024):
- \|Δ\| < 0.005 (do-nothing selective-correction anchor)

Loss decreased on training set.

## Watch signals (during training)

- **SI creeping up toward raw-field range (0.25+)**: bilateral-grid
  smoothness claim broken. Check grid resolution, guide computation, or
  whether the head has learned to route information via a non-guide
  path. Report immediately — this is the Exp 1-D pathology re-emerging
  and blocks Phase 2.
- **\|Δ\| collapsing to 0 on CVD-active images**: identity trap.
  Increase lr, reduce λ_tv, or check that FiLM MLP gradients are flowing.
- **ratio_w plateauing well below target** (e.g., P stalls at 1.03 by
  step 5000): loss frontier issue — likely λ_n dominating, or L_c/L_g
  ratio needs adjustment. Consider a small learning-rate warmup.
- **1761 \|Δ\|  > 0.005 during training**: the model is starting to
  correct where there's no confusion — do-nothing property broken. Check
  w computation and w·delta gating in `model.forward`.

## After a full-run PASS

- Best checkpoint is `model_step<N>_PASS.pt` in out_dir.
- Copy to a stable name: `cp outputs/v2_phase1/model_step<N>_PASS.pt \\
    outputs/v2_phase1/model_phase1_final.pt`
- Phase 2 (ONNX + browser): use `wrap_for_onnx(model, cvd_type)` per
  type and export three ONNX files (`_p.onnx`, `_d.onnx`, `_t.onnx`).
  Grid slicing is already gather-only (see `model.py::_slice_grid`).

## Files touched by Phase 1

- `cvdlens_v2/model.py` — architecture + ONNX-safe slicing
- `cvdlens_v2/losses.py` — `contrast_loss` (Phase 0 unchanged; excess
  param exposed as opt-in)
- `cvdlens_v2/train.py` — training loop + resume + inline val
- `cvdlens_v2/test_model.py` — unit + smoke tests
- `cvdlens_v2/kaggle_train.py` — Kaggle path resolver + defaults
- `cvdlens_v2/validate_multi.py` — 10-image bank (reused for val)
- `outputs/v2_phase1/RUNBOOK.md` — this file
