"""
Unit + smoke tests for CVDCorrectionNet.

Run: py -m cvdlens_v2.test_model
"""

from __future__ import annotations
from pathlib import Path

import sys
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.model import (
    CVDCorrectionNet, grid_tv, wrap_for_onnx, CVD_TYPES,
)
from cvdlens_v2.basis import get_confusion_dir
from cvdlens_v2.color import srgb_to_linear


def _seed(x=0):
    torch.manual_seed(x)


# ─────────────────────────────────────────────────────────────────────
# Unit test (a): w = 0  ⇒  out == orig  (identity when no confusion)
# ─────────────────────────────────────────────────────────────────────
def test_zero_w_is_identity():
    """
    Use a pure-gray input so the CVD sim ≈ input → ΔE_Lab ≈ 0 → w ≈ 0.
    Pipeline should return the input unchanged even with a random-init
    grid head (though our zero-init head trivially satisfies this).

    To make the test independent of head init, we deliberately non-zero
    the head bias so the grid produces nonzero d_lum/d_c, then check that
    w gating still drives out ≈ orig.
    """
    _seed(0)
    net = CVDCorrectionNet(pretrained_backbone=False)
    # Poison head bias so the grid is nonzero even at test time
    with torch.no_grad():
        net.grid_head.bias.fill_(0.1)

    orig = torch.full((1, 3, 128, 128), 0.5)  # solid gray
    for t in CVD_TYPES:
        r = net(orig, cvd_type=t, severity=1.0)
        w_max = r["w"].max().item()
        max_abs_diff = (r["out_srgb"] - orig).abs().max().item()
        # w ≈ 0 on solid gray → residual delta*w tiny; allow small numerical slack
        assert w_max < 1e-3, f"[{t}] expected w≈0 on gray, got w_max={w_max:.2e}"
        assert max_abs_diff < 1e-4, \
            f"[{t}] out - orig max|Δ| = {max_abs_diff:.2e} > 1e-4"
    print("[OK] test_zero_w_is_identity — solid-gray input passes through")


# ─────────────────────────────────────────────────────────────────────
# Unit test (b): dot(delta, confusion_dir) ≈ 0  (visible-subspace guarantee)
# ─────────────────────────────────────────────────────────────────────
def test_delta_orthogonal_to_confusion_line():
    """
    delta = d_lum·b_L + d_c·b_C is built in the visible subspace, so its
    projection onto the confusion direction must be numerically zero for
    every pixel — regardless of how the grid was trained.
    """
    _seed(1)
    net = CVDCorrectionNet(pretrained_backbone=False)
    # Non-trivial head so delta is nonzero
    with torch.no_grad():
        net.grid_head.weight.normal_(0, 0.5)
        net.grid_head.bias.normal_(0, 0.5)

    orig = torch.rand(2, 3, 96, 96)
    for t in CVD_TYPES:
        r = net(orig, cvd_type=t, severity=1.0)
        conf = get_confusion_dir(t, device=orig.device,
                                 dtype=orig.dtype).view(1, 3, 1, 1)
        comp = (r["delta"] * conf).sum(dim=1)      # (B, H, W)
        max_abs = comp.abs().max().item()
        assert max_abs < 1e-5, \
            f"[{t}] max |delta·conf_dir| = {max_abs:.2e} > 1e-5"
    print("[OK] test_delta_orthogonal_to_confusion_line — visible-subspace check")


# ─────────────────────────────────────────────────────────────────────
# Unit test (c): uniform patch  ⇒  delta variance in patch ≪ image variance
#                (bilateral-grid smoothness — Phase 1's answer to Exp 1-D)
# ─────────────────────────────────────────────────────────────────────
def test_uniform_patch_delta_smooth():
    """
    Feed A) a uniform-brightness, uniform-color image and B) a
    random-brightness image through the same random-init grid. The
    bilateral-grid smoothness claim is:

        var(delta | uniform input) ≪ var(delta | random input)

    Because a uniform input has a constant luminance guide → every pixel
    samples the same D-slice → only *spatial* trilinear interpolation
    contributes to delta variance (Lipschitz across H_g×W_g bins). A
    random input samples random D-slices per pixel → uncorrelated
    per-pixel jitter dominates variance.

    This is the structural claim that replaces the failed loss-level
    excess penalty (see phase0_final_report.md § Exp 1-D). If it holds,
    the bilateral grid's inductive smoothness is what suppresses speckle
    that raw-field optimization cannot.
    """
    _seed(2)
    net = CVDCorrectionNet(pretrained_backbone=False).eval()
    # Non-trivial random grid so delta is nonzero
    with torch.no_grad():
        net.grid_head.weight.normal_(0, 0.5)
        net.grid_head.bias.normal_(0, 0.5)

    H = W = 256
    uniform = torch.full((1, 3, H, W), 0.6)
    random = torch.rand(1, 3, H, W)

    for t in CVD_TYPES:
        with torch.no_grad():
            r_u = net(uniform, cvd_type=t, severity=1.0)
            r_r = net(random, cvd_type=t, severity=1.0)
        v_u = r_u["delta"].var(dim=(2, 3)).sum().item()
        v_r = r_r["delta"].var(dim=(2, 3)).sum().item()
        ratio = v_u / (v_r + 1e-12)
        assert ratio < 0.05, \
            f"[{t}] var(delta|uniform)/var(delta|random) = {ratio:.4f} " \
            "(want < 0.05). Bilateral-grid smoothness claim FAILED — " \
            "guide is leaking pixel-level variance into delta."
    print("[OK] test_uniform_patch_delta_smooth — bilateral-grid smoothness holds")


# ─────────────────────────────────────────────────────────────────────
# Smoke: forward → loss → backward (graph connectivity, gradient flow)
# ─────────────────────────────────────────────────────────────────────
def smoke_forward_backward():
    from cvdlens_v2.losses import CVDLossV2

    _seed(3)
    net = CVDCorrectionNet(pretrained_backbone=False).train()
    orig = torch.rand(2, 3, 128, 128, requires_grad=False)
    orig_lin = srgb_to_linear(orig)
    loss_fn = CVDLossV2(use_lpips=False)

    r = net(orig, cvd_type="p", severity=1.0)
    L_tv = grid_tv(r["grid"])
    total, comps = loss_fn(
        out_linear=r["out_linear"], orig_linear=orig_lin,
        out_srgb=r["out_srgb"], orig_srgb=orig,
        w=r["w"], cvd_type="p", severity=1.0,
    )
    total = total + 0.03 * L_tv
    total.backward()

    # Check that head + FiLM MLP + backbone last-used layer received grads
    def _grad_ok(p):
        return p.grad is not None and torch.isfinite(p.grad).all() \
               and p.grad.abs().max().item() > 0

    checks = {
        "grid_head.weight": _grad_ok(net.grid_head.weight),
        "film_mlp[-1].weight": _grad_ok(net.film_mlp[-1].weight),
    }
    assert all(checks.values()), f"Missing/zero grads: {checks}"
    print(f"[OK] smoke_forward_backward — total={total.item():.4f}  "
          f"comps={ {k: round(v, 4) for k, v in comps.items()} }")


# ─────────────────────────────────────────────────────────────────────
# Smoke: ONNX STATIC-shape export (opset 16) + onnxruntime parity check
# ─────────────────────────────────────────────────────────────────────
def smoke_onnx_export_static(tmp_path: Path):
    """
    Export at fixed (1, 3, 256, 256). No dynamic_axes. Then verify
    PyTorch vs onnxruntime output matches (atol < 1e-3). If this fails
    — especially on grid_sample 5D — Phase 1 stops before training: the
    slicing needs redesign, and finding out post-train would force a
    re-run.
    """
    _seed(4)
    net = CVDCorrectionNet(pretrained_backbone=False).eval()
    wrapped = wrap_for_onnx(net, cvd_type="p").eval()

    dummy_srgb = torch.rand(1, 3, 256, 256)
    dummy_sev = torch.tensor([[1.0]])
    dummy_w = torch.rand(1, 1, 256, 256)

    with torch.no_grad():
        pt_out = wrapped(dummy_srgb, dummy_sev, dummy_w)

    out_path = tmp_path / "cvdcorrection_p_static.onnx"
    export_kwargs = dict(
        input_names=["srgb", "severity", "w"],
        output_names=["out_srgb"],
        opset_version=16,
        # STATIC — no dynamic_axes on any input/output
    )
    # Prefer legacy tracer if new dynamo pipeline is default (avoids
    # Dim.DYNAMIC inference clashes on static inputs).
    try:
        torch.onnx.export(wrapped, (dummy_srgb, dummy_sev, dummy_w),
                          str(out_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch: no dynamo kwarg
        torch.onnx.export(wrapped, (dummy_srgb, dummy_sev, dummy_w),
                          str(out_path), **export_kwargs)
    size_kb = out_path.stat().st_size / 1024
    print(f"[export] wrote {out_path.name} ({size_kb:.0f} KB)")

    # Parity check via onnxruntime
    try:
        import onnxruntime as ort
    except ImportError as e:
        print(f"[WARN] smoke_onnx_export_static — onnxruntime not installed: {e}. "
              "Export succeeded; parity check SKIPPED.")
        return

    sess = ort.InferenceSession(str(out_path),
                                providers=["CPUExecutionProvider"])
    ort_out = sess.run(["out_srgb"], {
        "srgb": dummy_srgb.numpy(),
        "severity": dummy_sev.numpy(),
        "w": dummy_w.numpy(),
    })[0]
    diff = (torch.from_numpy(ort_out) - pt_out).abs().max().item()
    assert diff < 1e-3, \
        f"ONNX↔PyTorch parity FAILED: max|diff|={diff:.2e}"
    print(f"[OK] smoke_onnx_export_static — opset 16 static export + "
          f"ORT parity max|diff|={diff:.2e}")


def main():
    print("=" * 62)
    print("CVDCorrectionNet — unit + smoke tests")
    print("=" * 62)
    test_zero_w_is_identity()
    test_delta_orthogonal_to_confusion_line()
    test_uniform_patch_delta_smooth()
    smoke_forward_backward()
    tmp = Path(__file__).parent / "_tmp_test_artifacts"
    tmp.mkdir(exist_ok=True)
    smoke_onnx_export_static(tmp)
    print("\nALL UNIT TESTS PASSED (smokes non-fatal on WARN)")


if __name__ == "__main__":
    main()
