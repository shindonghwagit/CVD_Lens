"""
작업 3 — 채택된 w 정의(W1b)로 손실 영향 시뮬레이션 (학습 전 마지막 검증, 재학습 없음).

현행 t 체크포인트(배포 cvdlens_t.onnx = step 9000)의 출력에서, L_contrast의 혼동영역
기여가 W0(현행 w) 대비 W1b(Brettel+(12,30)) w로 실제로 커지는지 확인. 커지면 재학습 시
모델이 파랑을 움직일 유인(gradient)이 생긴다는 근거.

L_contrast = mean_s( m_s · relu(C_o - C_a)^2 ), m = w. 같은 출력(=같은 C_o,C_a)에 w만
바꿔 재계산하므로 차이는 순수하게 w(게이팅) 효과다.

Run: py -m cvdlens_v2.diag_tritan_loss
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvdlens_v2 import infer_local as il
from cvdlens_v2.color import srgb_to_linear, rgb_to_lab
from cvdlens_v2.simulation import simulate
from cvdlens_v2.losses import contrast_loss, _contrast_magnitude, _downsample
from cvdlens_v2.diag_tritan_w import w_variant, masks, tennis_proxy, T_RETUNE
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

OUT = Path("reports/tritan_w_diagnosis"); OUT.mkdir(parents=True, exist_ok=True)


def load256(path_or_arr):
    if isinstance(path_or_arr, np.ndarray):
        img = path_or_arr
    else:
        img = cv2.cvtColor(cv2.imread(str(path_or_arr)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    return img


def analyze(name, img_np):
    orig_lin = srgb_to_linear(torch.from_numpy(img_np).permute(2, 0, 1)[None])
    out256 = il._run_float(img_np, "t", 1.0)                 # current t model (ONNX)
    out_lin = srgb_to_linear(torch.from_numpy(np.clip(out256, 0, 1)).permute(2, 0, 1)[None])
    sim_out = simulate(out_lin, "t", 1.0)
    lab_orig = rgb_to_lab(orig_lin.float())
    lab_sim_out = rgb_to_lab(sim_out.float())

    # scale-1 deficit map (the core L_c integrand, unweighted)
    C_o = _contrast_magnitude(lab_orig)
    C_a = _contrast_magnitude(lab_sim_out)
    deficit2 = torch.relu(C_o - C_a).pow(2)                  # (1,1,256,256)

    conf, achr = masks(img_np, "t")
    conf_t = torch.from_numpy(conf)[None, None]

    rec = {"name": name}
    for v in ("W0", "W1b"):
        w = w_variant(orig_lin, "t", v)
        Lc = float(contrast_loss(lab_orig, lab_sim_out, w))
        # confusion-region weighted contribution (scale 1)
        contrib = (w * deficit2)
        conf_contrib = float(contrib[conf_t].sum())
        total_contrib = float(contrib.sum() + 1e-9)
        rec[v] = dict(L_contrast=round(Lc, 5),
                      conf_region_weighted_deficit=round(conf_contrib, 4),
                      conf_share=round(conf_contrib / total_contrib, 3),
                      w_conf_mean=round(float(w[conf_t].mean()) if conf.sum() else float('nan'), 3))
    rec["Lc_ratio_W1b_over_W0"] = round(rec["W1b"]["L_contrast"] / (rec["W0"]["L_contrast"] + 1e-9), 2)
    rec["conf_deficit_ratio_W1b_over_W0"] = round(
        rec["W1b"]["conf_region_weighted_deficit"] / (rec["W0"]["conf_region_weighted_deficit"] + 1e-9), 2)
    print(f"[{name}] L_c W0={rec['W0']['L_contrast']} → W1b={rec['W1b']['L_contrast']} "
          f"(×{rec['Lc_ratio_W1b_over_W0']})  | conf weighted-deficit ×{rec['conf_deficit_ratio_W1b_over_W0']} "
          f"| conf share {rec['W0']['conf_share']}→{rec['W1b']['conf_share']}")
    return rec


def main():
    imgs = {"blue_sea": load256("outputs/artifact_analysis/blue_sea.jpg"),
            "tennis_proxy": load256(tennis_proxy())}
    if Path("outputs/daily_test/traffic_street.jpg").exists():
        imgs["traffic_street"] = load256("outputs/daily_test/traffic_street.jpg")
    recs = [analyze(n, im) for n, im in imgs.items()]
    (OUT / "loss_impact.json").write_text(
        json.dumps({"t_retune": list(T_RETUNE), "results": recs}, indent=2), encoding="utf-8")
    print(f"\n[save] {OUT/'loss_impact.json'}")
    return recs


if __name__ == "__main__":
    main()
