"""
Phase 2 · Step 1 — export model_best (step 9000) to browser-deployable ONNX.

One self-contained graph per CVD type (P/D/T). Each graph:
    inputs  : srgb (1,3,256,256) float32, severity (1,1) float32
    output  : out_srgb (1,3,256,256) float32
    static shapes, opset 16, no dynamic_axes.

Severity is a LIVE input (not frozen): it drives both the FiLM conditioning and
the Machado confusion-weight matrix (see model.wrap_for_onnx). The confusion
weight is computed inside the graph, so the browser feeds only (srgb, severity).

Usage:
    py -m cvdlens_v2.export_onnx \
        --ckpt outputs/phase3_kaggle/v2_phase1/model_best.pt \
        --out-dir outputs/v2_phase2
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch

from cvdlens_v2.model import CVDCorrectionNet, wrap_for_onnx, CVD_TYPES

SIZE = 256


def load_model(ckpt: str, device="cpu") -> CVDCorrectionNet:
    net = CVDCorrectionNet(pretrained_backbone=False).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    print(f"[ckpt] {ckpt}  step={ck.get('step')}")
    return net


def export_one(net, cvd_type, out_path):
    wrapped = wrap_for_onnx(net, cvd_type).eval()
    dummy_srgb = torch.rand(1, 3, SIZE, SIZE)
    dummy_sev = torch.tensor([[1.0]])

    # Severity-liveness spot check: 0.5 vs 1.0 must differ (FiLM + Machado both live)
    with torch.no_grad():
        o10 = wrapped(dummy_srgb, torch.tensor([[1.0]]))
        o05 = wrapped(dummy_srgb, torch.tensor([[0.5]]))
    sev_delta = (o10 - o05).abs().max().item()

    kw = dict(input_names=["srgb", "severity"], output_names=["out_srgb"],
              opset_version=16)
    try:
        torch.onnx.export(wrapped, (dummy_srgb, dummy_sev), str(out_path),
                          dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(wrapped, (dummy_srgb, dummy_sev), str(out_path), **kw)

    size_kb = out_path.stat().st_size / 1024
    print(f"[{cvd_type}] {out_path.name}  {size_kb:.0f} KB  "
          f"severity 0.5-vs-1.0 max|Δout|={sev_delta:.4f} "
          f"{'OK (live)' if sev_delta > 1e-4 else 'FROZEN?!'}")
    return {"type": cvd_type, "file": out_path.name, "size_kb": round(size_kb, 1),
            "severity_live_delta": sev_delta}


def main(a):
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = load_model(a.ckpt)

    rows = []
    for t in CVD_TYPES:
        rows.append(export_one(net, t, out_dir / f"cvdlens_{t}.onnx"))

    print("\n── export summary ──")
    for r in rows:
        flag = "live" if r["severity_live_delta"] > 1e-4 else "FROZEN"
        print(f"  cvdlens_{r['type']}.onnx  {r['size_kb']:>6.1f} KB  severity={flag}")
    all_live = all(r["severity_live_delta"] > 1e-4 for r in rows)
    all_small = all(r["size_kb"] < 1024 for r in rows)
    print(f"  all severity-live: {all_live}   all < 1 MB: {all_small}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",
                   default="outputs/phase3_kaggle/v2_phase1/model_best.pt")
    p.add_argument("--out-dir", default="outputs/v2_phase2")
    main(p.parse_args())
