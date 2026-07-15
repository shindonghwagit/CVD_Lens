"""
CVDCorrectionNet — Phase 1 training (single stage).

Loss:
    CVDLossV2(λ_c=1.0, λ_g=0.15, λ_n=0.15) + λ_tv · grid_tv(grid_5d)

    λ_tv is applied to the bilateral grid COEFFICIENTS (H_g×W_g axes),
    not the sliced full-res field. Cheap, semantically equivalent (grid_sample
    is Lipschitz-1). Value kept at 0.03 (Phase 0 confirmed) as insurance
    against noisy grid coefficients early in training; the architecture
    itself provides the smoothness prior that Exp 1-D's loss-level excess
    penalty could not.

Validation:
    Every --val-every epochs, run the model on the 10-image COCO val bank
    (same as Phase 0's validate_multi). Apply RECALIBRATED Phase 1
    thresholds (below), NOT Phase 0's raw-field numbers — see phase0_final_report
    § "Confirmed training config" for why:

        ratio_w    P ≥ 1.10   D ≥ 1.13   T ≥ 1.27    (tv=0.1 row − 0.03)
        SI < 0.15                                     (network target)
        |Δ| > 0.01 for 9 CVD-active images
        |Δ| < 0.005 for 000000001761 (do-nothing anchor)

Usage:
    py -m cvdlens_v2.train --smoke                       # 10 iters, sanity
    py -m cvdlens_v2.train --data-dir C:/Users/SCH/coco/train2017 \\
        --iters 20000 --batch-size 8 --lr 2e-4 --lambda-tv 0.03
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.model import CVDCorrectionNet, grid_tv, CVD_TYPES
from cvdlens_v2.losses import CVDLossV2
from cvdlens_v2.color import srgb_to_linear, linear_to_srgb
from cvdlens_v2.confusion import compute_confusion_weight, _gaussian_kernel_2d
from cvdlens_v2.simulation import simulate
from cvdlens_v2.validate_loss import (
    _blur_rgb, _cvd_contrast_ratio_w, _speckle_index,
)
from cvdlens_v2.validate_multi import IMAGES as VAL_BANK, LOW_W_STEM


# ── Phase 1 success criteria (see docstring) ────────────────────────
RATIO_W_MIN_BY_TYPE = {"p": 1.10, "d": 1.13, "t": 1.27}
SI_MAX = 0.15
DELTA_MIN = 0.01
DELTA_MAX_LOW_W = 0.005


# ── Dataset ─────────────────────────────────────────────────────────
class COCOCropDataset(torch.utils.data.Dataset):
    """Random 256×256 crop from any *.jpg under a directory."""
    def __init__(self, root: str, crop: int = 256):
        self.paths = sorted(Path(root).glob("*.jpg"))
        assert self.paths, f"No .jpg found under {root}"
        self.crop = crop
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        w, h = img.size
        # Resize so shorter side ≥ crop, then random crop
        scale = self.crop / min(w, h)
        if scale > 1.0:
            new_w = int(w * scale + 0.5)
            new_h = int(h * scale + 0.5)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            w, h = new_w, new_h
        x0 = random.randint(0, w - self.crop)
        y0 = random.randint(0, h - self.crop)
        img = img.crop((x0, y0, x0 + self.crop, y0 + self.crop))
        return self.to_tensor(img)     # (3, crop, crop) sRGB [0,1]


# ── Validation ──────────────────────────────────────────────────────
@torch.no_grad()
def _load_val_image(stem: str, size: int, device) -> torch.Tensor:
    path = Path(f"C:/Users/SCH/coco/val2017/{stem}.jpg")
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).to(device)


@torch.no_grad()
def validate(model: CVDCorrectionNet, device, size: int = 256) -> dict:
    """Run 10-image bank × 3 types. Apply Phase 1 thresholds. Return summary."""
    model.eval()
    rows = []
    all_pass = True
    for stem in VAL_BANK:
        is_low = (stem == LOW_W_STEM)
        orig = _load_val_image(stem, size, device)
        orig_lin = srgb_to_linear(orig)
        for t in CVD_TYPES:
            r = model(orig, cvd_type=t, severity=1.0)
            delta_mag = (r["out_srgb"] - orig).abs().mean().item()
            # ratio_w on w-weighted CVD-view blur (same as Phase 0 metric)
            sim_out = simulate(r["out_linear"], t, 1.0)
            sim_orig = simulate(orig_lin, t, 1.0)
            rw = _cvd_contrast_ratio_w(sim_out, sim_orig, r["w"],
                                        blur_sigma=1.0)
            si = _speckle_index(r["delta"], r["w"], sigma=2.0, ksize=9)

            if is_low:
                id_ok = delta_mag < DELTA_MAX_LOW_W
                rw_ok = True     # skip contrast on low-w
                si_ok = True     # skip SI on low-w (delta≈0 anyway)
            else:
                id_ok = delta_mag > DELTA_MIN
                rw_ok = rw > RATIO_W_MIN_BY_TYPE[t]
                si_ok = si < SI_MAX
            passed = id_ok and rw_ok and si_ok
            all_pass = all_pass and passed
            rows.append({
                "stem": stem, "type": t, "is_low_w": is_low,
                "delta": delta_mag, "ratio_w": rw, "SI": si,
                "pass_identity": id_ok, "pass_ratio_w": rw_ok,
                "pass_SI": si_ok, "pass": passed,
            })
    model.train()
    return {"all_pass": all_pass, "rows": rows}


def _print_val_summary(v: dict):
    by_stem = {}
    for r in v["rows"]:
        by_stem.setdefault(r["stem"], []).append(r)
    for stem, rs in by_stem.items():
        low = rs[0]["is_low_w"]
        tag = "  (LOW-W)" if low else ""
        line = " | ".join(
            f"{r['type']}: |Δ|={r['delta']:.4f} rw={r['ratio_w']:.3f} "
            f"SI={r['SI']:.3f} [{'P' if r['pass'] else 'F'}]"
            for r in rs
        )
        print(f"  {stem}{tag}  {line}")
    print(f"  ALL PASS: {v['all_pass']}")


# ── Training loop ───────────────────────────────────────────────────
def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: lr={args.lr}  batch={args.batch_size}  "
          f"λ_c=1.0  λ_g=0.15  λ_n=0.15  λ_tv={args.lambda_tv}  "
          f"iters={args.iters}  crop={args.crop}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    if args.smoke:
        # Fake dataset: 8 random 256×256 images (no COCO needed)
        print("[smoke] Using synthetic 8-image dataset")
        imgs = torch.rand(8, 3, args.crop, args.crop)
        class Wrap(torch.utils.data.Dataset):
            def __len__(self): return len(imgs)
            def __getitem__(self, i): return imgs[i]
        ds = Wrap()
    else:
        ds = COCOCropDataset(args.data_dir, crop=args.crop)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True, persistent_workers=(args.workers > 0),
    )
    steps_per_epoch = len(loader)

    # Model + loss + opt
    model = CVDCorrectionNet(pretrained_backbone=args.pretrained).to(device)
    if torch.cuda.device_count() > 1 and args.data_parallel:
        model = nn.DataParallel(model)
        print(f"[DataParallel] using {torch.cuda.device_count()} GPUs")
    loss_fn = CVDLossV2(lambda_c=1.0, lambda_g=0.15, lambda_n=0.15,
                        use_lpips=args.use_lpips).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    total_steps = args.iters if args.iters > 0 else args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=total_steps, eta_min=args.lr / 10)

    step = 0
    epoch = 0
    t_start = time.time()
    history = []

    # ── Resume ─────────────────────────────────────────────────────
    if args.resume:
        # Load latest checkpoint (highest step number) matching model_step*.pt
        cks = sorted(out_dir.glob("model_step*.pt"))
        if cks:
            latest = cks[-1]
            print(f"[resume] loading {latest.name}")
            # weights_only=False required: checkpoint contains argparse.Namespace
            # and history dicts. Default flipped to True in PyTorch 2.6.
            ck = torch.load(latest, map_location=device, weights_only=False)
            (model.module if isinstance(model, nn.DataParallel) else model)\
                .load_state_dict(ck["state_dict"])
            if "optim" in ck:
                optim.load_state_dict(ck["optim"])
            if "scheduler" in ck:
                scheduler.load_state_dict(ck["scheduler"])
            step = ck.get("step", 0)
            history = ck.get("history", [])
            print(f"[resume] resumed from step {step}")
        else:
            print("[resume] no checkpoint found — starting fresh")
    while step < total_steps:
        epoch += 1
        for batch in loader:
            if step >= total_steps:
                break
            batch = batch.to(device)
            # Random CVD type per batch (uniform sampling)
            cvd_type = random.choice(CVD_TYPES)

            r = model(batch, cvd_type=cvd_type, severity=1.0)
            orig_lin = srgb_to_linear(batch)
            total, comps = loss_fn(
                out_linear=r["out_linear"], orig_linear=orig_lin,
                out_srgb=r["out_srgb"], orig_srgb=batch,
                w=r["w"], cvd_type=cvd_type, severity=1.0,
            )
            L_tv = grid_tv(r["grid"])
            total = total + args.lambda_tv * L_tv

            optim.zero_grad(set_to_none=True)
            total.backward()
            optim.step()
            scheduler.step()

            step += 1
            if step % args.log_every == 0 or step == 1:
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_start
                print(f"[step {step:>6d}/{total_steps}]  type={cvd_type}  "
                      f"total={total.item():.4f}  L_c={comps['L_contrast']:.3f}  "
                      f"L_g={comps['L_global']:.3f}  L_n={comps['L_natural']:.3f}  "
                      f"L_tv={L_tv.item():.4f}  lr={lr_now:.2e}  "
                      f"[{elapsed:.0f}s]")
                history.append({
                    "step": step, "type": cvd_type,
                    "total": total.item(), **comps, "L_tv": L_tv.item(),
                })

            # Val
            if step % args.val_every == 0 or step == total_steps:
                print(f"\n[step {step}] VALIDATION")
                v = validate(model, device)
                _print_val_summary(v)
                # Save
                (out_dir / f"val_step{step:06d}.json").write_text(
                    json.dumps(v, indent=2))
                if v["all_pass"]:
                    ck = out_dir / f"model_step{step:06d}_PASS.pt"
                else:
                    ck = out_dir / f"model_step{step:06d}.pt"
                sd = (model.module if isinstance(model, nn.DataParallel)
                      else model).state_dict()
                torch.save({
                    "step": step,
                    "state_dict": sd,
                    "optim": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": vars(args),
                    "history": history,
                }, ck)
                # Also write "latest.pt" symlink-like copy for easy resume
                latest_ck = out_dir / "model_latest.pt"
                torch.save({
                    "step": step,
                    "state_dict": sd,
                    "optim": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": vars(args),
                    "history": history,
                }, latest_ck)
                print(f"  saved {ck.name}  +  model_latest.pt")
                print()

    print(f"\nTraining finished at step {step} in {time.time() - t_start:.0f}s")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="C:/Users/SCH/coco/train2017")
    p.add_argument("--out-dir", type=str,
                   default="C:/Users/SCH/graduation_project/outputs/v2_phase1")
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--iters", type=int, default=20000,
                   help="Total optimizer steps (takes precedence over --epochs)")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--lambda-tv", type=float, default=0.03)
    p.add_argument("--pretrained", action="store_true",
                   help="Load ImageNet-pretrained mobilenet weights")
    p.add_argument("--use-lpips", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="10 steps on 8 synthetic images (no COCO needed)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest model_step*.pt in --out-dir")
    p.add_argument("--data-parallel", action="store_true",
                   help="Wrap model in nn.DataParallel if >1 CUDA device")
    args = p.parse_args()
    if args.smoke:
        args.iters = 10
        args.val_every = 10
        args.log_every = 1
        args.batch_size = 2
    main(args)
