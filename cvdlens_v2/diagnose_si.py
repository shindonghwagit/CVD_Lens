"""
SI decomposition diagnostic — is SI blowing up because delta genuinely
has speckle, or because the ratio's denominator is tiny (guide-aligned
smooth-but-varying delta)?

Reports per (image × cvd_type):
    SI       = mean(w·|δ − G_σ(δ)|) / (mean(w·|δ|) + ε)      (original)
    SI_abs   = mean(w·|δ − G_σ(δ)|)                          (numerator only)
    delta_mag= mean(w·|δ|)                                    (denominator)
    corr_guide = corr( |δ − G_σ(δ)|,  |∇luma(orig)| )         within w>0.3
                 — if > 0.5, the "high frequency" tracks luminance edges,
                 i.e. delta variation is *guide-aligned edge structure*,
                 not random speckle.
    SI_uniform = SI computed only in "uniform-original" pixels
                 (w>0.3 AND |∇luma|<τ). If SI_uniform ≪ SI_full, the
                 pathology is at edges (guide response), not in flat
                 regions — confirms non-speckle interpretation.

Usage:
    py -m cvdlens_v2.diagnose_si \\
        --ckpt C:/Users/SCH/Downloads/model_latest.zip \\
        --stems 000000000724,000000000632,000000001584,000000001761
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.model import CVDCorrectionNet
from cvdlens_v2.color import srgb_to_linear
from cvdlens_v2.confusion import _gaussian_kernel_2d


def _blur_channels(x, sigma=2.0, ksize=9):
    C = x.shape[1]
    k = _gaussian_kernel_2d(sigma, ksize, x.device, x.dtype).expand(
        C, 1, ksize, ksize).contiguous()
    pad = ksize // 2
    return F.conv2d(F.pad(x, [pad] * 4, mode="reflect"), k, groups=C)


def _luma(rgb_lin):
    return (0.2126 * rgb_lin[:, 0:1]
            + 0.7152 * rgb_lin[:, 1:2]
            + 0.0722 * rgb_lin[:, 2:3])


def _grad_mag(x):
    dy = F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 0, 0, 1))
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    return (dy.pow(2) + dx.pow(2)).sqrt()


def _corr(a, b, mask):
    """Pearson corr within mask (bool). Both a, b: (H*W,)."""
    m = mask.reshape(-1) > 0
    if m.sum() < 32:
        return float("nan")
    ax = a.reshape(-1)[m] - a.reshape(-1)[m].mean()
    bx = b.reshape(-1)[m] - b.reshape(-1)[m].mean()
    denom = (ax.pow(2).mean().sqrt() * bx.pow(2).mean().sqrt() + 1e-12)
    return ((ax * bx).mean() / denom).item()


@torch.no_grad()
def diagnose_one(net, orig_srgb, cvd_type):
    r = net(orig_srgb, cvd_type=cvd_type, severity=1.0)
    delta = r["delta"]                                    # (1, 3, H, W)
    w = r["w"]                                            # (1, 1, H, W)

    # SI numerator/denominator
    d_blur = _blur_channels(delta, sigma=2.0, ksize=9)
    hi = (delta - d_blur).abs().mean(dim=1, keepdim=True)  # (1,1,H,W)
    mag = delta.abs().mean(dim=1, keepdim=True)

    numer = (w * hi).mean().item()
    denom = (w * mag).mean().item()
    si = numer / (denom + 1e-8)

    # Guide-alignment: correlate hi with |∇luma(orig)| within w>0.3
    orig_lin = srgb_to_linear(orig_srgb)
    luma = _luma(orig_lin)
    luma_grad = _grad_mag(luma)
    mask_hi = (w > 0.3).squeeze()
    corr = _corr(hi.squeeze(), luma_grad.squeeze(), mask_hi)

    # SI in uniform regions only (w>0.3 AND |∇luma|<0.02)
    lg = luma_grad.squeeze()
    tau = 0.02       # linear-luma gradient magnitude threshold
    m_uniform = ((w.squeeze() > 0.3) & (lg < tau)).unsqueeze(0).unsqueeze(0).float()
    if m_uniform.sum() > 32:
        n_u = (m_uniform * hi).mean().item()
        d_u = (m_uniform * mag).mean().item()
        si_uniform = n_u / (d_u + 1e-8)
        pix_u = int(m_uniform.sum().item())
    else:
        si_uniform = float("nan")
        pix_u = 0

    return {
        "SI": si,
        "SI_abs": numer,
        "delta_mag": denom,
        "corr_guide": corr,
        "SI_uniform": si_uniform,
        "uniform_pixels": pix_u,
        "w_mean": w.mean().item(),
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    net = CVDCorrectionNet(pretrained_backbone=False).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    print(f"[ckpt] step={ck.get('step')}   device={device}")

    print()
    print(f"{'stem':<14} {'type':<5}  {'w̄':>6}  {'SI':>7}  {'SI_abs':>9}"
          f"  {'|δ|':>9}  {'corr':>7}  {'SI_uni':>7}  {'#uni':>6}")
    print("-" * 92)

    for stem in [s.strip() for s in args.stems.split(",") if s.strip()]:
        img = Image.open(f"{args.val_dir}/{stem}.jpg").convert("RGB").resize(
            (args.size, args.size), Image.LANCZOS)
        orig = transforms.ToTensor()(img).unsqueeze(0).to(device)
        for t in ["p", "d", "t"]:
            d = diagnose_one(net, orig, t)
            print(f"{stem:<14} {t:<5}  {d['w_mean']:>6.3f}  "
                  f"{d['SI']:>7.3f}  {d['SI_abs']:>9.6f}  {d['delta_mag']:>9.6f}"
                  f"  {d['corr_guide']:>7.3f}  {d['SI_uniform']:>7.3f}"
                  f"  {d['uniform_pixels']:>6d}")
        print()

    print("interpretation guide:")
    print("  - SI 큰데 SI_abs ~0 & |δ| ~0  → 비율 폭주 (지표 아티팩트, 실제 speckle 없음)")
    print("  - corr_guide > 0.5           → hi가 luma edge에 정렬됨 (guide 경로 정상, speckle 아님)")
    print("  - SI_uniform ≪ SI            → 균일-original 영역엔 speckle 없음")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-dir", default="C:/Users/SCH/coco/val2017")
    p.add_argument("--stems",
                   default="000000000724,000000000632,000000001584,000000001761")
    p.add_argument("--size", type=int, default=256)
    main(p.parse_args())
