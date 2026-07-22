"""
Phase 2 · Step 1 — PyTorch ↔ onnxruntime parity + CPU inference benchmark.

Per CVD type, compare model_best (PyTorch forward) against the exported ONNX
graph on:
    3 random inputs + 2 real val images (724, 1584)   × severity {0.5, 1.0}
    = 10 cases per type.
Pass criterion: max|diff| < 1e-3 (scaffold hit 1.6e-7; anything > 1e-4 is
reported as a concern). Also benches ORT CPU single-image latency (median of
20 runs after 5 warmup). Writes parity_report.json.

Usage:
    py -m cvdlens_v2.parity_check --onnx-dir outputs/v2_phase2 \
        --ckpt outputs/phase3_kaggle/v2_phase1/model_best.pt
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from cvdlens_v2.model import CVDCorrectionNet, CVD_TYPES

SIZE = 256
REAL_STEMS = ["000000000724", "000000001584"]
SEVERITIES = [0.5, 1.0]
PASS_TOL = 1e-3
CONCERN_TOL = 1e-4


def load_real(stem, val_dir, device):
    img = Image.open(f"{val_dir}/{stem}.jpg").convert("RGB").resize(
        (SIZE, SIZE), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).to(device)


def main(a):
    device = "cpu"
    torch.manual_seed(a.seed)
    net = CVDCorrectionNet(pretrained_backbone=False).to(device)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ck["state_dict"]); net.eval()

    # input set: 3 random + 2 real
    inputs = [("rand0", torch.rand(1, 3, SIZE, SIZE)),
              ("rand1", torch.rand(1, 3, SIZE, SIZE)),
              ("rand2", torch.rand(1, 3, SIZE, SIZE))]
    for st in REAL_STEMS:
        inputs.append((st, load_real(st, a.val_dir, device)))

    report = {"ckpt_step": ck.get("step"), "tol_pass": PASS_TOL,
              "per_type": {}, "cases": []}
    overall_max = 0.0

    for t in CVD_TYPES:
        onnx_path = Path(a.onnx_dir) / f"cvdlens_{t}.onnx"
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])
        tmax = 0.0
        for name, srgb in inputs:
            for sev in SEVERITIES:
                with torch.no_grad():
                    pt = net(srgb, cvd_type=t, severity=sev)["out_srgb"].numpy()
                onnx_out = sess.run(["out_srgb"], {
                    "srgb": srgb.numpy().astype(np.float32),
                    "severity": np.array([[sev]], dtype=np.float32)})[0]
                d = float(np.abs(onnx_out - pt).max())
                tmax = max(tmax, d); overall_max = max(overall_max, d)
                report["cases"].append(
                    {"type": t, "input": name, "severity": sev,
                     "max_abs_diff": d})
        report["per_type"][t] = {"max_abs_diff": tmax,
                                  "pass": tmax < PASS_TOL}
        print(f"[{t}] max|diff| over 10 cases = {tmax:.2e}  "
              f"{'PASS' if tmax < PASS_TOL else 'FAIL'}")

    report["overall_max_abs_diff"] = overall_max
    report["all_pass"] = overall_max < PASS_TOL
    report["concern"] = overall_max >= CONCERN_TOL

    # ── benchmark: ORT CPU single-image latency ──
    print("\n── ORT CPU latency (256×256, 1 image; 5 warmup + 20 timed) ──")
    bench = {}
    dummy = np.random.rand(1, 3, SIZE, SIZE).astype(np.float32)
    sev1 = np.array([[1.0]], dtype=np.float32)
    for t in CVD_TYPES:
        sess = ort.InferenceSession(str(Path(a.onnx_dir) / f"cvdlens_{t}.onnx"),
                                    providers=["CPUExecutionProvider"])
        feed = {"srgb": dummy, "severity": sev1}
        for _ in range(5):
            sess.run(["out_srgb"], feed)
        ts = []
        for _ in range(20):
            s = time.perf_counter(); sess.run(["out_srgb"], feed)
            ts.append((time.perf_counter() - s) * 1000)
        med = statistics.median(ts)
        bench[t] = {"median_ms": round(med, 2),
                    "min_ms": round(min(ts), 2), "max_ms": round(max(ts), 2)}
        print(f"  cvdlens_{t}: median {med:.1f} ms  (min {min(ts):.1f} / max {max(ts):.1f})")
    report["benchmark_cpu_ort"] = bench

    print(f"\nOVERALL max|diff| = {overall_max:.2e}   "
          f"ALL PASS (<{PASS_TOL:g}): {report['all_pass']}   "
          f"concern (>={CONCERN_TOL:g}): {report['concern']}")

    dst = Path(a.onnx_dir) / "parity_report.json"
    dst.write_text(json.dumps(report, indent=2))
    print(f"wrote {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx-dir", default="outputs/v2_phase2")
    p.add_argument("--ckpt",
                   default="outputs/phase3_kaggle/v2_phase1/model_best.pt")
    p.add_argument("--val-dir", default="C:/Users/SCH/coco/val2017")
    p.add_argument("--seed", type=int, default=20260722)
    main(p.parse_args())
