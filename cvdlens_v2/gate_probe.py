"""
Investigation #2 (w-gate selectivity) + #3 (tritan global shift).

#2: does the confusion weight w suppress |delta| on achromatic / low-saturation
    pixels? Per type p/d/t. Also native-res bleed from delta upsampling near a
    saturated↔achromatic boundary.
#3: tritan global desaturation on blue sea — is it because blue IS the tritan
    confusion colour (high w by design, same basis logic as p/d), not a bug?

w is recomputed with cvdlens_v2.confusion.compute_confusion_weight — the SAME
Lab-ΔE-threshold-blur formula the ONNX graph runs internally (wrap_for_onnx).
Simulation for CRR/NP uses Brettel-1997 (cvdlens_v2.simulation, method='brettel').
NO daltonize import. MEASURE ONLY.

Run: py -m cvdlens_v2.gate_probe
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.artifact_probe import correct, _load, OUT, _letterbox
from cvdlens_v2.color import srgb_to_linear, rgb_to_lab
from cvdlens_v2.confusion import compute_confusion_weight
from cvdlens_v2.simulation import simulate
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

TYPES = ["p", "d", "t"]


def w_map(lb256, cvd_type):
    lin = srgb_to_linear(torch.from_numpy(lb256).permute(2, 0, 1)[None].float())
    return compute_confusion_weight(lin, cvd_type, 1.0)[0, 0].numpy()


def saturation(img):
    mx = img.max(2); mn = img.min(2)
    return (mx - mn) / (mx + 1e-6)


# ── #2: selectivity ──────────────────────────────────────────────────────
def selectivity(name):
    img = _load(OUT / f"{name}.jpg")
    lb, _ = _letterbox(img, 256)
    sat = saturation(lb)
    achro = sat < 0.12            # near-achromatic (white/gray/reflections)
    satur = sat > 0.5            # saturated
    rows = {}
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for j, t in enumerate(TYPES):
        r = correct(img, t, 1.0)
        dmag = np.linalg.norm(r["delta256"], axis=2)
        w = w_map(lb, t)
        rows[t] = dict(
            achro_deltamed=float(np.median(dmag[achro])) if achro.any() else None,
            achro_delta_p90=float(np.percentile(dmag[achro], 90)) if achro.any() else None,
            satur_deltamed=float(np.median(dmag[satur])) if satur.any() else None,
            achro_w_med=float(np.median(w[achro])) if achro.any() else None,
            satur_w_med=float(np.median(w[satur])) if satur.any() else None,
            corr_w_delta=float(np.corrcoef(w.ravel(), dmag.ravel())[0, 1]),
        )
        # scatter w vs |delta|
        idx = np.random.default_rng(0).choice(dmag.size, 4000, replace=False)
        ax[0, j].scatter(w.ravel()[idx], dmag.ravel()[idx], s=2, alpha=0.2)
        ax[0, j].set_title(f"{t}: corr(w,|Δ|)={rows[t]['corr_w_delta']:.2f}")
        ax[0, j].set_xlabel("w"); ax[0, j].set_ylabel("|Δ|@256")
        # |delta| distributions achro vs satur
        ax[1, j].hist(dmag[achro], bins=40, alpha=0.6, density=True, label=f"achro (n={achro.sum()})")
        if satur.any():
            ax[1, j].hist(dmag[satur], bins=40, alpha=0.6, density=True, label=f"satur (n={satur.sum()})")
        ax[1, j].axvline(rows[t]['achro_deltamed'], color="C0", ls="--")
        ax[1, j].set_title(f"{t}: achro |Δ| med={rows[t]['achro_deltamed']:.3f} vs satur {rows[t]['satur_deltamed']:.3f}")
        ax[1, j].set_xlabel("|Δ|@256"); ax[1, j].legend(fontsize=7)
    fig.suptitle(f"[#2 selectivity] {name} — w-gating on achromatic (sat<0.12) vs saturated (sat>0.5)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f"gate2_{name}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[#2 {name}]", json.dumps(rows, indent=1))
    return {name: rows}


# ── #3: tritan on blue sea ───────────────────────────────────────────────
def tritan_analysis(name="blue_sea"):
    img = _load(OUT / f"{name}.jpg")
    lb, _ = _letterbox(img, 256)
    lin = srgb_to_linear(torch.from_numpy(lb).permute(2, 0, 1)[None].float())
    rows = {}
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for j, t in enumerate(TYPES):
        r = correct(img, t, 1.0)
        out_lin = srgb_to_linear(torch.from_numpy(np.clip(r["out256"], 0, 1)).permute(2, 0, 1)[None].float())
        w = compute_confusion_weight(lin, t, 1.0)
        # CRR (confusion-weighted contrast ratio, Brettel view) + NP (|Δ| sRGB)
        sim_o = simulate(lin, t, 1.0, "brettel")
        sim_c = simulate(out_lin, t, 1.0, "brettel")
        lab_o, lab_c = rgb_to_lab(sim_o), rgb_to_lab(sim_c)
        def cmag(lab):
            dy = lab[:, :, 1:] - lab[:, :, :-1]; dx = lab[:, :, :, 1:] - lab[:, :, :, :-1]
            gy = (dy**2).sum(1, keepdim=True); gx = (dx**2).sum(1, keepdim=True)
            g = torch.zeros_like(lab[:, :1]); g[:, :, 1:] += gy; g[:, :, :, 1:] += gx
            return torch.sqrt(g + 1e-6)
        wsum = w.sum() + 1e-6
        crr = float((( cmag(lab_c) * w).sum() / wsum) / (((cmag(lab_o) * w).sum() / wsum) + 1e-6))
        np_delta = float(np.abs(np.clip(r["out256"], 0, 1) - lb).mean())
        rows[t] = dict(w_mean=float(w.mean()), w_median=float(w.median()),
                       crr=crr, np_delta=np_delta,
                       delta_mean=float(np.linalg.norm(r["delta256"], axis=2).mean()))
        ax[0, j].imshow(w[0, 0].numpy(), cmap="viridis", vmin=0, vmax=1)
        ax[0, j].set_title(f"{t}: w (mean={rows[t]['w_mean']:.2f})"); ax[0, j].axis("off")
        ax[1, j].imshow(np.clip(r["out256"], 0, 1))
        ax[1, j].set_title(f"{t}: corrected  NP|Δ|={np_delta:.3f}  CRR={crr:.2f}"); ax[1, j].axis("off")
    fig.suptitle(f"[#3 tritan] {name} — per-type w & correction. Blue=tritan confusion colour ⇒ high w by design", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f"gate3_{name}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[#3 {name}]", json.dumps(rows, indent=1))
    return {name: rows}


def main():
    out = {"selectivity": {}, "tritan": {}}
    for n in ["stop_sign", "blue_sea"]:
        out["selectivity"].update(selectivity(n))
    out["tritan"].update(tritan_analysis("blue_sea"))
    (OUT / "gate_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[save] {OUT/'gate_stats.json'}")


if __name__ == "__main__":
    main()
