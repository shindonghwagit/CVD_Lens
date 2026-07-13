"""Why does daltonize FAIL the sanity gate? Print raw C values."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.color import srgb_to_linear, rgb_to_lab
from cvdlens_v2.confusion import compute_confusion_weight
from cvdlens_v2.losses import _contrast_magnitude
from cvdlens_v2.simulation import simulate, daltonize


IMAGE = 'C:/Users/SCH/coco/val2017/000000000724.jpg'
SIZE = 256


def load_image():
    img = Image.open(IMAGE).convert('RGB').resize((SIZE, SIZE), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0)


def diagnose(cvd_type: str, orig_srgb: torch.Tensor):
    print(f"\n{'='*70}\n[{cvd_type}] C(orig) vs C(sim(orig)) vs C(sim(daltonize))\n{'='*70}")
    orig_lin = srgb_to_linear(orig_srgb)
    w = compute_confusion_weight(orig_lin, cvd_type, 1.0)
    sim_orig = simulate(orig_lin, cvd_type, 1.0)
    dalt_lin = daltonize(orig_lin, cvd_type, 1.0)
    sim_dalt = simulate(dalt_lin, cvd_type, 1.0)

    lab_orig = rgb_to_lab(orig_lin)
    lab_sim_orig = rgb_to_lab(sim_orig)
    lab_sim_dalt = rgb_to_lab(sim_dalt)

    C_o = _contrast_magnitude(lab_orig)
    C_so = _contrast_magnitude(lab_sim_orig)
    C_sd = _contrast_magnitude(lab_sim_dalt)

    # Where w > 0.5 (confusion regions)
    conf_mask = (w > 0.5).float()
    n_conf = conf_mask.sum().item()

    # Averages
    print(f"  Overall averages:")
    print(f"    C(orig)         mean = {C_o.mean().item():.3f}")
    print(f"    C(sim(orig))    mean = {C_so.mean().item():.3f}")
    print(f"    C(sim(dalt))    mean = {C_sd.mean().item():.3f}")
    print(f"  In confusion regions (w>0.5, n={int(n_conf)} pixels):")
    for name, C in [("C(orig)", C_o), ("C(sim(orig))", C_so), ("C(sim(dalt))", C_sd)]:
        w_avg = (C * conf_mask).sum().item() / (n_conf + 1e-8)
        print(f"    {name:15s} in confusion = {w_avg:.3f}")

    # Deficit stats
    def_id = torch.relu(C_o - C_so)       # deficit at identity
    def_dt = torch.relu(C_o - C_sd)       # deficit at daltonize
    print(f"\n  Deficit distribution (relu(C_orig - C_sim_out)):")
    print(f"    identity:  mean={def_id.mean().item():.3f}  q95={def_id.flatten().quantile(0.95).item():.3f}  max={def_id.max().item():.3f}")
    print(f"    daltonize: mean={def_dt.mean().item():.3f}  q95={def_dt.flatten().quantile(0.95).item():.3f}  max={def_dt.max().item():.3f}")

    # w-weighted squared deficit (what L_c actually integrates)
    ws_id = (w * def_id.pow(2)).mean().item()
    ws_dt = (w * def_dt.pow(2)).mean().item()
    print(f"    L_c(identity)  contribution ≈ {ws_id:.4f}")
    print(f"    L_c(daltonize) contribution ≈ {ws_dt:.4f}   (should be lower!)")

    # WHERE is daltonize losing? Show top pixels with biggest INCREASE in deficit
    delta_def = def_dt - def_id
    top_bad = delta_def[0, 0].flatten().topk(10)
    print(f"\n  Top-10 pixels where daltonize INCREASED deficit vs identity:")
    for rank, (val, idx) in enumerate(zip(top_bad.values.tolist(), top_bad.indices.tolist())):
        y, x = idx // SIZE, idx % SIZE
        c_o_here = C_o[0, 0, y, x].item()
        c_so_here = C_so[0, 0, y, x].item()
        c_sd_here = C_sd[0, 0, y, x].item()
        w_here = w[0, 0, y, x].item()
        print(f"    #{rank+1} at ({y},{x}) w={w_here:.2f}  "
              f"C_orig={c_o_here:.2f}  C_sim_orig={c_so_here:.2f}  C_sim_dalt={c_sd_here:.2f}  "
              f"Δdeficit=+{val:.3f}")


def main():
    orig_srgb = load_image()
    for t in ["p", "d", "t"]:
        diagnose(t, orig_srgb)


if __name__ == "__main__":
    main()
