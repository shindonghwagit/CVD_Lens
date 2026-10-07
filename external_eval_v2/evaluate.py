"""Evaluate deployed protan/deutan models on the frozen v2 manifest."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_eval"))
import evaluate as base  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
TYPES = ("p", "d")
SIMULATORS = ("machado", "brettel")
BRETTEL = {
    kind: base.LMS2RGB @ matrix @ base.RGB2LMS
    for kind, matrix in {
        "p": np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]], np.float32),
        "d": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]], np.float32),
    }.items()
}


def simulate(x: np.ndarray, kind: str, method: str) -> np.ndarray:
    matrix = base.MACHADO[kind] if method == "machado" else BRETTEL[kind]
    return np.clip(np.einsum("ij,hwj->hwi", matrix, x), 0, 1)


def confusion_weight(x: np.ndarray, kind: str, method: str) -> np.ndarray:
    delta = np.linalg.norm(base.lab(x) - base.lab(simulate(x, kind, method)), axis=2)
    weight = np.clip((delta - 2.0) / 10.0, 0, 1).astype(np.float32)
    return cv2.GaussianBlur(weight, (11, 11), 3, borderType=cv2.BORDER_REFLECT)


def crr(orig: np.ndarray, out: np.ndarray, weight: np.ndarray, kind: str, method: str) -> float:
    if float(weight.mean()) < 1e-6:
        return float("nan")
    before = cv2.GaussianBlur(simulate(orig, kind, method), (5, 5), 1, borderType=cv2.BORDER_REFLECT)
    after = cv2.GaussianBlur(simulate(out, kind, method), (5, 5), 1, borderType=cv2.BORDER_REFLECT)

    def gradient(x: np.ndarray) -> np.ndarray:
        dx = np.pad(np.abs(x[:, 1:] - x[:, :-1]).mean(2), ((0, 0), (0, 1)))
        dy = np.pad(np.abs(x[1:] - x[:-1]).mean(2), ((0, 1), (0, 0)))
        return dx + dy

    denominator = float((weight * gradient(before)).sum() / (weight.sum() + 1e-8))
    numerator = float((weight * gradient(after)).sum() / (weight.sum() + 1e-8))
    return numerator / (denominator + 1e-8)


def bootstrap_ci(values: list[float], seed: int = 20261007) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(10000, len(array)), replace=True), axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    dataset = Path(__file__).resolve().parent
    with (dataset / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    OUT.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []

    for index, item in enumerate(manifest, 1):
        original = base.load_image(dataset / "images" / item["filename"])
        original_linear = base.srgb_linear(original)
        for kind in TYPES:
            corrected = base.deployed._correct_image(original, kind, 1.0)
            corrected_linear = base.srgb_linear(corrected)
            delta = np.abs(corrected - original)
            for method in SIMULATORS:
                weight = confusion_weight(original_linear, kind, method)
                rows.append({
                    "id": item["id"], "category": item["category"], "cvd_type": kind,
                    "simulator": method, "w_mean": float(weight.mean()),
                    "crr": crr(original_linear, corrected_linear, weight, kind, method),
                    "np_mae": float(delta.mean()),
                    "large_change_ratio": float((delta.max(2) > 0.1).mean()),
                })
        if index % 10 == 0 or index == len(manifest):
            print(f"[{index}/{len(manifest)}]", flush=True)

    with (OUT / "eval_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result: dict[str, object] = {"dataset": {"images": len(manifest), "categories": 20}, "types": {}}
    for kind in TYPES:
        result["types"][kind] = {}
        for method in SIMULATORS:
            subset = [row for row in rows if row["cvd_type"] == kind and row["simulator"] == method]
            cutoff = float(np.median([float(row["w_mean"]) for row in subset]))
            meaningful = [row for row in subset if float(row["w_mean"]) >= cutoff and np.isfinite(float(row["crr"]))]
            values = [float(row["crr"]) for row in meaningful]
            categories: dict[str, list[float]] = defaultdict(list)
            for row in meaningful:
                categories[str(row["category"])].append(float(row["crr"]))
            result["types"][kind][method] = {
                "meaningful_n": len(values), "w_median_cutoff": cutoff,
                "crr_mean": float(np.mean(values)), "crr_median": float(np.median(values)),
                "crr_mean_95ci": bootstrap_ci(values),
                "failures_below_1": int(sum(value < 1 for value in values)),
                "success_rate": float(np.mean(np.asarray(values) >= 1)),
                "np_mae_mean": float(np.mean([float(row["np_mae"]) for row in subset])),
                "large_change_ratio_mean": float(np.mean([float(row["large_change_ratio"]) for row in subset])),
                "category_crr_mean": {key: float(np.mean(value)) for key, value in sorted(categories.items())},
            }

    (OUT / "eval_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# P/D external validation v2", "", f"Frozen independent set: {len(manifest)} accepted images across 20 everyday-life strata.", "",
             "| Type | Simulator | Relevant n | CRR mean (95% CI) | median | failures | success | NP MAE | large change |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in TYPES:
        for method in SIMULATORS:
            value = result["types"][kind][method]
            lo, hi = value["crr_mean_95ci"]
            lines.append(f"| {kind.upper()} | {method} | {value['meaningful_n']} | {value['crr_mean']:.3f} ({lo:.3f}-{hi:.3f}) | {value['crr_median']:.3f} | {value['failures_below_1']} | {value['success_rate']:.1%} | {value['np_mae_mean']:.4f} | {value['large_change_ratio_mean']:.1%} |")
    lines += ["", "CRR > 1 indicates increased local contrast in simulator-defined confusion regions. Results are engineering evidence, not a clinical user study."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
