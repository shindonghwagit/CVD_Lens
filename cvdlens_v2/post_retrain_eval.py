"""
작업 3 — 재학습 후 평가 원커맨드 (재학습 전 baseline도 동일 스크립트로 산출).

지금(재학습 전) 돌리면 현행 t ONNX의 'before' 기준선, 재학습+export 후 --model-dir로
새 ONNX를 가리켜 'after'를 뽑아 비교한다. 지표 정의는 기존 스크립트 재사용.

이미지: blue_sea, tennis_proxy(합성), traffic_street, skin_portrait.
산출(이미지별): 원본/보정(t)/w맵(t,신 confusion.py)/sim(t)원본/sim(t)보정 몽타주 + 지표표
(ΔE00 mean, 파랑영역 ΔE00, w_t mean, CRR).

Run(before): py -m cvdlens_v2.post_retrain_eval
Run(after) : py -m cvdlens_v2.post_retrain_eval --model-dir <새 onnx 폴더> --tag after
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import onnxruntime as rt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvdlens_v2 import infer_local as il
from cvdlens_v2.color import srgb_to_linear, linear_to_srgb
from cvdlens_v2.confusion import compute_confusion_weight
from cvdlens_v2.simulation import simulate
from cvdlens_v2.daily_test import sat, crr
from cvdlens_v2.artifact_probe import ciede2000, to_lab
from cvdlens_v2.diag_tritan_w import tennis_proxy, masks
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

OUT = Path("reports/tritan_retrain_eval"); OUT.mkdir(parents=True, exist_ok=True)


def load(path_or_arr, cap=512):
    if isinstance(path_or_arr, np.ndarray):
        img = path_or_arr
    else:
        img = cv2.cvtColor(cv2.imread(str(path_or_arr)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = img.shape[:2]; m = max(h, w)
    if m > cap:
        s = cap / m; img = cv2.resize(img, (round(w * s), round(h * s)))
    return img


def blue_region_dE(de, img_np):
    conf, _ = masks(img_np, "t")            # blue/yellow confusion hues
    return float(de[conf].mean()) if conf.sum() else float("nan")


def sim_disp(lin):
    return linear_to_srgb(lin)[0].permute(1, 2, 0).clamp(0, 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(il.MODEL_DIR),
                    help="dir with cvdlens_{p,d,t}.onnx (default = current deployed)")
    ap.add_argument("--tag", default="before")
    args = ap.parse_args()

    # point infer_local at the requested model dir (default = current)
    md = Path(args.model_dir)
    il.SESSIONS = {t: rt.InferenceSession(str(md / f"cvdlens_{t}.onnx"),
                                          providers=["CPUExecutionProvider"]) for t in ("p", "d", "t")}
    print(f"[{args.tag}] model dir: {md}")

    images = {"blue_sea": load("outputs/artifact_analysis/blue_sea.jpg"),
              "tennis_proxy": load(tennis_proxy()),
              "traffic_street": load("outputs/daily_test/traffic_street.jpg"),
              "skin_portrait": load("outputs/daily_test/skin_portrait.jpg")}

    rows = []
    for name, img in images.items():
        lin = srgb_to_linear(torch.from_numpy(img).permute(2, 0, 1)[None])
        r = il.correct(img, "t", 1.0, use_guided=True)          # deployed path (guided ON)
        corr = r["corrected"]
        de = ciede2000(to_lab(img), to_lab(corr))
        w_t = compute_confusion_weight(lin, "t", 1.0)[0, 0].numpy()   # new confusion.py
        rec = dict(name=name, tag=args.tag,
                   deE_mean=round(float(de.mean()), 2),
                   blue_region_deE=round(blue_region_dE(de, img), 2),
                   w_t_mean=round(float(w_t.mean()), 3),
                   crr=round(crr(img, corr, "t"), 3),
                   sat_delta=round(float(sat(corr).mean() - sat(img).mean()), 4))
        rows.append(rec)
        # montage
        so = sim_disp(simulate(lin, "t", 1.0)); sc = sim_disp(simulate(srgb_to_linear(
            torch.from_numpy(corr).permute(2, 0, 1)[None]), "t", 1.0))
        fig, ax = plt.subplots(1, 5, figsize=(24, 5))
        for a, (ti, im) in zip(ax, [("original", img), ("corrected t", corr),
                                    (f"w_t (mean {rec['w_t_mean']})", w_t),
                                    ("sim(t) orig", so), ("sim(t) corr", sc)]):
            cmap = "viridis" if ti.startswith("w_t") else None
            im_ = a.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            a.set_title(ti, fontsize=10); a.axis("off")
        fig.suptitle(f"[{args.tag}] {name} — tritan  ΔE00={rec['deE_mean']} blueΔE={rec['blue_region_deE']} "
                     f"w_t={rec['w_t_mean']} CRR={rec['crr']}", fontsize=12)
        fig.tight_layout(); fig.savefig(OUT / f"{name}_t_{args.tag}.png", dpi=82, bbox_inches="tight"); plt.close(fig)
        print(f"  {name}: ΔE00={rec['deE_mean']} blueΔE={rec['blue_region_deE']} "
              f"w_t={rec['w_t_mean']} CRR={rec['crr']}")

    (OUT / f"eval_{args.tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[save] {OUT/('eval_'+args.tag+'.json')}")
    return rows


if __name__ == "__main__":
    main()
