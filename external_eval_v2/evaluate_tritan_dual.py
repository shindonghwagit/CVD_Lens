"""Independent Brettel/Machado evaluation of the production T rule."""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_eval"))
import evaluate as base  # noqa: E402
sys.path.insert(0, str(ROOT / "cvd-lens" / "inference"))
import main  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
MATRICES = {"brettel": main._TRITAN_SIM, "machado": main._TRITAN_MACHADO}
THRESHOLDS = {"brettel": (12.0, 30.0), "machado": (5.0, 25.0)}


def simulate(x, matrix):
    return np.clip(np.einsum("ij,hwj->hwi", matrix, x), 0, 1)


def weight(x, matrix, method):
    de = np.linalg.norm(base.lab(x) - base.lab(simulate(x, matrix)), axis=2)
    lo, hi = THRESHOLDS[method]
    w = np.clip((de - lo) / (hi - lo), 0, 1).astype(np.float32)
    return cv2.GaussianBlur(w, (11, 11), 3, borderType=cv2.BORDER_REFLECT)


def crr(orig, out, w, matrix):
    before = cv2.GaussianBlur(simulate(orig, matrix), (5, 5), 1, borderType=cv2.BORDER_REFLECT)
    after = cv2.GaussianBlur(simulate(out, matrix), (5, 5), 1, borderType=cv2.BORDER_REFLECT)
    def grad(x):
        dx = np.pad(np.abs(x[:, 1:] - x[:, :-1]).mean(2), ((0, 0), (0, 1)))
        dy = np.pad(np.abs(x[1:] - x[:-1]).mean(2), ((0, 1), (0, 0)))
        return dx + dy
    den = float((w * grad(before)).sum() / (w.sum() + 1e-8))
    num = float((w * grad(after)).sum() / (w.sum() + 1e-8))
    return num / (den + 1e-8)


def main_eval():
    with (HERE / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))
    indexes = defaultdict(int); rows = []
    for i, item in enumerate(manifest, 1):
        split = "dev" if indexes[item["category"]] % 2 == 0 else "test"
        indexes[item["category"]] += 1
        orig = base.load_image(HERE / "images" / item["filename"])
        out = main._correct_image(orig, "t", 1.0)
        orig_lin, out_lin = base.srgb_linear(orig), base.srgb_linear(out)
        delta = np.abs(out - orig)
        for method, matrix in MATRICES.items():
            w = weight(orig_lin, matrix, method)
            rows.append({"id": item["id"], "category": item["category"], "split": split,
                         "simulator": method, "w_mean": float(w.mean()),
                         "crr": crr(orig_lin, out_lin, w, matrix),
                         "np_mae": float(delta.mean()),
                         "large_change_ratio": float((delta.max(2) > .1).mean())})
        if i % 10 == 0 or i == len(manifest): print(f"[{i}/{len(manifest)}]", flush=True)
    summary = {}
    for split in ("dev", "test"):
        summary[split] = {}
        for method in MATRICES:
            subset = [r for r in rows if r["split"] == split and r["simulator"] == method]
            cutoff = float(np.median([r["w_mean"] for r in subset]))
            relevant = [r for r in subset if r["w_mean"] >= cutoff]
            vals = np.array([r["crr"] for r in relevant])
            summary[split][method] = {"n": len(vals), "crr_mean": float(vals.mean()),
                "crr_median": float(np.median(vals)), "failures_below_0.9995": int((vals < .9995).sum()),
                "success_rate": float((vals >= .9995).mean()),
                "np_mae": float(np.mean([r["np_mae"] for r in subset])),
                "large_change_ratio": float(np.mean([r["large_change_ratio"] for r in subset]))}
    passed = all(summary["test"][m]["success_rate"] >= .95 for m in MATRICES)
    result = {"pass_threshold": "test success >=95% on both simulators", "passed": passed,
              "summary": summary, "rows": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / "tritan_dual_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "summary": summary}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__": main_eval()
