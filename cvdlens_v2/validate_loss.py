"""
Phase 0 validation — v2: low-res field + LR schedule + monotonic loss check.

No network. d_lum/d_c are LOW-RES fields (default 64x64) upsampled bilinearly
to match the input — this mimics the smoothness inductive bias of the future
bilateral grid model and eliminates speckle artifacts of full-res per-pixel
parametrization.

Pass criteria (all must hold per CVD type):
    1. |optimized - orig| > 0.02   (identity escape)
    2. final_total_loss < initial_total_loss   (loss actually decreased)
    3. blur-guarded contrast ratio > 1.15   (measurable perceptual gain,
       robust to pixel-level speckle)

Usage:
    py -m cvdlens_v2.validate_loss --image C:/Users/SCH/coco/val2017/000000000724.jpg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.basis import compose_delta, get_confusion_dir
from cvdlens_v2.color import srgb_to_linear, linear_to_srgb, rgb_to_lab
from cvdlens_v2.confusion import compute_confusion_weight, _blur, _gaussian_kernel_2d
from cvdlens_v2.losses import CVDLossV2
from cvdlens_v2.simulation import simulate


def _blur_rgb(x: torch.Tensor, sigma: float = 1.0, ksize: int = 5) -> torch.Tensor:
    """Gaussian blur on 3-channel RGB via depthwise conv."""
    C = x.shape[1]
    k = _gaussian_kernel_2d(sigma, ksize, x.device, x.dtype)  # (1,1,k,k)
    k = k.expand(C, 1, ksize, ksize).contiguous()
    pad = ksize // 2
    return F.conv2d(F.pad(x, [pad] * 4, mode='reflect'), k, groups=C)


CVD_NAMES = {"p": "Protanopia", "d": "Deuteranopia", "t": "Tritanopia"}


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0)   # (1,3,H,W) sRGB [0,1]


def _upsample(field: torch.Tensor, size: int) -> torch.Tensor:
    """(B, 1, h, w) low-res field → (B, 1, size, size) bilinear."""
    return F.interpolate(field, size=(size, size), mode='bilinear', align_corners=False)


def _cvd_contrast_ratio(sim_out: torch.Tensor, sim_orig: torch.Tensor,
                        blur_sigma: float = 1.0) -> float:
    """
    Ratio of gradient magnitude between CVD-view of output and input.
    Both are Gaussian-blurred first (σ=1) to suppress pixel-level speckle so
    the metric reflects genuine perceptual contrast, not noise.
    """
    a = _blur_rgb(sim_out, sigma=blur_sigma, ksize=5)
    o = _blur_rgb(sim_orig, sigma=blur_sigma, ksize=5)
    gy_o = (o[:, :, 1:] - o[:, :, :-1]).abs().mean()
    gy_a = (a[:, :, 1:] - a[:, :, :-1]).abs().mean()
    gx_o = (o[:, :, :, 1:] - o[:, :, :, :-1]).abs().mean()
    gx_a = (a[:, :, :, 1:] - a[:, :, :, :-1]).abs().mean()
    return ((gy_a + gx_a) / (gy_o + gx_o + 1e-8)).item()


def _init_fields(direction: str, beta: float,
                 B: int, field_size: int, device) -> tuple:
    """Constant-scalar init of low-res d_lum/d_c along visible-basis axis."""
    zero = torch.zeros(B, 1, field_size, field_size, device=device)
    if direction in (None, "", "identity"):
        return zero.clone(), zero.clone()
    if direction == "+b_L":
        return torch.full_like(zero, +beta), zero.clone()
    if direction == "-b_L":
        return torch.full_like(zero, -beta), zero.clone()
    if direction == "+b_C":
        return zero.clone(), torch.full_like(zero, +beta)
    if direction == "-b_C":
        return zero.clone(), torch.full_like(zero, -beta)
    raise ValueError(f"Unknown init direction: {direction}")


def optimize(
    orig_srgb: torch.Tensor,
    cvd_type: str,
    severity: float = 1.0,
    n_iters: int = 400,
    lr: float = 0.005,
    field_size: int = 64,
    use_lpips: bool = False,
    log_every: int = 20,
    init_direction: str = "identity",
    init_beta: float = 0.0,
) -> dict:
    device = orig_srgb.device
    B, _, H, W = orig_srgb.shape

    orig_lin = srgb_to_linear(orig_srgb)
    w = compute_confusion_weight(orig_lin, cvd_type, severity)   # (B,1,H,W), no grad

    # LOW-RES learnable fields, warm-startable via init_direction/init_beta
    d_lum_init, d_c_init = _init_fields(init_direction, init_beta, B, field_size, device)
    d_lum = nn.Parameter(d_lum_init)
    d_c = nn.Parameter(d_c_init)

    optimizer = torch.optim.Adam([d_lum, d_c], lr=lr)
    # Cosine to lr/10 (not 0) so late iters can still refine — with cosine-to-zero
    # optimizer freezes after ~60% of iters and locks in a shallow minimum.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iters, eta_min=lr / 10,
    )
    loss_fn = CVDLossV2(use_lpips=use_lpips).to(device)

    # Fixed generator: makes loss & logging comparable across iters
    gen = torch.Generator(device=device).manual_seed(42)

    initial_total = None
    history = []
    for it in range(n_iters):
        optimizer.zero_grad()

        d_lum_up = _upsample(d_lum, H)
        d_c_up = _upsample(d_c, W)
        delta = compose_delta(d_lum_up, d_c_up, cvd_type)
        out_lin = (orig_lin + w * delta).clamp(0.0, 1.0)
        out_srgb = linear_to_srgb(out_lin)

        # Re-seed generator each iter → same K pairs sampled every iter → monotonic curve
        gen.manual_seed(42)
        total, comps = loss_fn(
            out_linear=out_lin,
            orig_linear=orig_lin,
            out_srgb=out_srgb,
            orig_srgb=orig_srgb,
            w=w,
            cvd_type=cvd_type,
            severity=severity,
            generator=gen,
        )
        total.backward()
        optimizer.step()
        scheduler.step()

        if initial_total is None:
            initial_total = total.item()

        if it % log_every == 0 or it == n_iters - 1:
            with torch.no_grad():
                pred_diff = (out_srgb - orig_srgb).abs().mean().item()
                sim_out = simulate(out_lin, cvd_type, severity)
                sim_orig = simulate(orig_lin, cvd_type, severity)
                conf = get_confusion_dir(cvd_type, device=device).view(1, 3, 1, 1)
                conf_comp = ((w * delta) * conf).sum(dim=1).abs().max().item()
                ratio_blur = _cvd_contrast_ratio(sim_out, sim_orig, blur_sigma=1.0)
                lr_now = scheduler.get_last_lr()[0]
            history.append({
                "iter": it,
                "total": total.item(),
                **comps,
                "pred_diff": pred_diff,
                "conf_component": conf_comp,
                "cvd_contrast_ratio": ratio_blur,
                "lr": lr_now,
            })
            print(f"  it={it:>4d}  total={total.item():.4f}  "
                  f"L_c={comps['L_contrast']:.3f}  L_g={comps['L_global']:.3f}  "
                  f"L_n={comps['L_natural']:.3f}  |Δ|={pred_diff:.4f}  "
                  f"ratio(blur)={ratio_blur:.3f}  lr={lr_now:.2e}  "
                  f"|conf|={conf_comp:.1e}")

    with torch.no_grad():
        d_lum_up = _upsample(d_lum, H)
        d_c_up = _upsample(d_c, W)
        delta = compose_delta(d_lum_up, d_c_up, cvd_type)
        out_lin = (orig_lin + w * delta).clamp(0.0, 1.0)
        out_srgb = linear_to_srgb(out_lin)
        sim_orig = simulate(orig_lin, cvd_type, severity)
        sim_out = simulate(out_lin, cvd_type, severity)

    final_total = history[-1]['total']
    return {
        "orig_srgb": orig_srgb,
        "sim_orig_srgb": linear_to_srgb(sim_orig),
        "out_srgb": out_srgb,
        "sim_out_srgb": linear_to_srgb(sim_out),
        "w": w,
        "history": history,
        "initial_total": initial_total,
        "final_total": final_total,
    }


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def plot_result(result: dict, cvd_type: str, out_path: Path):
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.15)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_sim_in = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[0, 2])
    ax_out = fig.add_subplot(gs[1, 0])
    ax_sim_out = fig.add_subplot(gs[1, 1])
    ax_loss_main = fig.add_subplot(gs[1, 2])
    ax_loss_nat = fig.add_subplot(gs[2, 2])
    ax_meta = fig.add_subplot(gs[2, :2])

    ax_orig.imshow(_to_np(result['orig_srgb'])); ax_orig.set_title('Original', fontweight='bold')
    ax_sim_in.imshow(_to_np(result['sim_orig_srgb'])); ax_sim_in.set_title(f'CVD Sim (input) — {CVD_NAMES[cvd_type]}', fontweight='bold')
    ax_w.imshow(result['w'][0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1); ax_w.set_title('Confusion w', fontweight='bold')
    ax_out.imshow(_to_np(result['out_srgb'])); ax_out.set_title('Optimized Output', fontweight='bold')
    ax_sim_out.imshow(_to_np(result['sim_out_srgb'])); ax_sim_out.set_title('CVD Sim (output)', fontweight='bold')
    for ax in [ax_orig, ax_sim_in, ax_w, ax_out, ax_sim_out]:
        ax.set_xticks([]); ax.set_yticks([])

    hist = result['history']
    iters = [h['iter'] for h in hist]
    ax_loss_main.plot(iters, [h['total'] for h in hist], 'k-', label='total', linewidth=2)
    ax_loss_main.plot(iters, [h['L_contrast'] for h in hist], 'r-', label='L_c', linewidth=1.4)
    ax_loss_main.plot(iters, [h['L_global'] for h in hist], 'g-', label='L_g', linewidth=1.4)
    ax_loss_main.plot(iters, [h['L_natural'] for h in hist], 'purple', label='L_natural',
                      linewidth=1.4)
    ax_loss_main.axhline(y=result['initial_total'], color='gray', linestyle=':',
                         label='initial total', alpha=0.6)
    ax_loss_main.set_xlabel('iter'); ax_loss_main.set_ylabel('loss')
    ax_loss_main.legend(fontsize=8); ax_loss_main.grid(True, alpha=0.3)
    ax_loss_main.set_title('Loss (total, L_c, L_g, L_natural — separated)', fontsize=10)

    ax_loss_nat.plot(iters, [h['pred_diff'] for h in hist], 'orange', label='|Δ|')
    ax_loss_nat.plot(iters, [h['cvd_contrast_ratio'] - 1.0 for h in hist], 'teal',
                     label='ratio(blur) − 1')
    ax_loss_nat.axhline(y=0.02, color='orange', linestyle=':', alpha=0.6, label='|Δ| target')
    ax_loss_nat.axhline(y=0.15, color='teal', linestyle=':', alpha=0.6, label='ratio-1 target')
    ax_loss_nat.set_xlabel('iter'); ax_loss_nat.legend(fontsize=8)
    ax_loss_nat.grid(True, alpha=0.3); ax_loss_nat.set_title('Metrics: |Δ|, blur-ratio', fontsize=10)

    final = hist[-1]
    loss_decrease = result['initial_total'] - result['final_total']
    ax_meta.axis('off')
    meta = (
        f"CVD type: {CVD_NAMES[cvd_type]}    n_iters: {len(hist)}\n"
        f"Initial total loss: {result['initial_total']:.4f}\n"
        f"Final   total loss: {result['final_total']:.4f}   "
        f"Δloss = {loss_decrease:+.4f}   "
        f"({'DECREASED' if loss_decrease > 0 else 'INCREASED'})\n\n"
        f"|Δ| = {final['pred_diff']:.4f}    (pass if > 0.02)\n"
        f"CVD contrast ratio (blur σ=1): {final['cvd_contrast_ratio']:.3f}    (pass if > 1.15)\n"
        f"Confusion-line residual: {final['conf_component']:.1e}    (must be ~0)\n"
    )
    ax_meta.text(0.02, 0.98, meta, family='monospace', fontsize=10,
                 verticalalignment='top')

    fig.suptitle(f"β v2 Phase 0 — {CVD_NAMES[cvd_type]}", fontsize=14, fontweight='bold')
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close()


def main(args):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tag_str = f"  init={args.init_direction} β0={args.init_beta}"
    print(f"Config: field_size={args.field_size}  lr={args.lr}"
          f"  iters={args.iters}{tag_str}")

    orig_srgb = load_image(args.image, size=args.size).to(device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    tag = args.tag
    verdict = {}
    for cvd_type in types:
        print(f"\n=== {CVD_NAMES[cvd_type]} (severity={args.severity}) ===")
        result = optimize(
            orig_srgb=orig_srgb,
            cvd_type=cvd_type,
            severity=args.severity,
            n_iters=args.iters,
            lr=args.lr,
            field_size=args.field_size,
            use_lpips=args.use_lpips,
            log_every=args.log_every,
            init_direction=args.init_direction,
            init_beta=args.init_beta,
        )
        img_name = Path(args.image).stem
        out_path = out_dir / f"validate_{img_name}_{cvd_type}{tag}.png"
        plot_result(result, cvd_type, out_path)
        print(f"  Saved: {out_path}")

        final = result['history'][-1]
        pass_identity = final['pred_diff'] > 0.02
        pass_ratio = final['cvd_contrast_ratio'] > 1.15
        pass_loss = result['final_total'] < result['initial_total']
        verdict[cvd_type] = {
            "pred_diff": final['pred_diff'],
            "cvd_contrast_ratio": final['cvd_contrast_ratio'],
            "initial_total": result['initial_total'],
            "final_total": result['final_total'],
            "identity_escaped": pass_identity,
            "contrast_recovered": pass_ratio,
            "loss_decreased": pass_loss,
        }

    print("\n" + "=" * 68)
    print("VERDICT — Phase 0 v2 (3 criteria per type)")
    print("=" * 68)
    all_pass = True
    for t, v in verdict.items():
        marks = [
            f"[{'OK' if v['identity_escaped'] else 'X ' }] |Δ| = {v['pred_diff']:.4f}  (> 0.02)",
            f"[{'OK' if v['contrast_recovered'] else 'X ' }] ratio(blur) = {v['cvd_contrast_ratio']:.3f}  (> 1.15)",
            f"[{'OK' if v['loss_decreased'] else 'X ' }] loss {v['initial_total']:.4f} → {v['final_total']:.4f}  (Δ={v['initial_total']-v['final_total']:+.4f})",
        ]
        all_pass = all_pass and all([v['identity_escaped'], v['contrast_recovered'], v['loss_decreased']])
        print(f"\n{CVD_NAMES[t]}:")
        for m in marks:
            print(f"  {m}")

    print("\n" + ("*** PHASE 0 v2 PASSED ***" if all_pass
                  else "*** PHASE 0 v2 FAILED — tune before training ***"))

    import json as _json
    vpath = out_dir / f"verdict{tag}.json"
    with open(vpath, 'w', encoding='utf-8') as f:
        _json.dump({"config": {"init_direction": args.init_direction,
                                "init_beta": args.init_beta,
                                "lr": args.lr,
                                "iters": args.iters,
                                "field_size": args.field_size},
                    "verdict": {t: {k: float(v) if isinstance(v, (int, float)) else v
                                    for k, v in vd.items()}
                                for t, vd in verdict.items()},
                    "all_pass": all_pass}, f, indent=2)
    print(f"Verdict → {vpath}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, default="C:/Users/SCH/coco/val2017/000000000724.jpg")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--severity", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.02, help="tuned: 0.005 → identity, 0.05 → diverge, 0.02 middle")
    p.add_argument("--field-size", type=int, default=64,
                   help="low-res field spatial size (upsampled bilinearly to input)")
    p.add_argument("--log-every", type=int, default=40)
    p.add_argument("--use-lpips", action="store_true")
    p.add_argument("--out-dir", type=str, default="C:/Users/SCH/graduation_project/outputs/v2_phase0")
    p.add_argument("--types", type=str, default="p,d,t",
                   help="Comma-separated subset of CVD types to run")
    p.add_argument("--init-direction", type=str, default="identity",
                   choices=["identity", "+b_L", "-b_L", "+b_C", "-b_C"],
                   help="Warm-start direction for d_lum/d_c fields")
    p.add_argument("--init-beta", type=float, default=0.0,
                   help="Warm-start magnitude along --init-direction")
    p.add_argument("--tag", type=str, default="",
                   help="Suffix on output filenames (e.g. '_runA')")
    args = p.parse_args()
    main(args)
