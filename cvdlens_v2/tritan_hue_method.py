"""'우리 방식' tritan — 채도보존 hue 회전 (blue->보라 / yellow->주황) + guided 경계평활.
test-set(manifest) + 출력기반 지표로 채점. confusion.py 등 공용코드 불변, 실험 스크립트 전용.

지표(출력 기반, CRITERIA 의도를 방법 출력에 대응):
  커버리지: 파랑+노랑 각 카테고리 대상영역 보정ΔE mean>=4, min>=2  AND  CRR>=1.0
  일관성  : 파랑 카테고리 ΔE mean 간 (상대) — 참고 기록
  물빠짐  : 전 카테고리 |satΔ|<=0.03  (hue회전은 구조적으로 0 근처여야)
  선택성  : red/green/gray 이미지 전체 ΔE mean<=1.5 (거의 무변)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cvd-lens" / "inference"))
from guided import guided_filter
from cvdlens_v2.daily_test import sat, crr
from cvdlens_v2.artifact_probe import ciede2000, to_lab

MAN = json.load(open("cvdlens_v2/testsets/tritan_blue/manifest.json"))
VAL = MAN["coco_val_dir"]
OUT = Path("reports/tritan_gate_eval"); OUT.mkdir(parents=True, exist_ok=True)
SIZE = 256
DEG = 26.0


def resolve(p): return f"{VAL}/{p[5:]}" if p.startswith("coco:") else p
def load(p):
    img = cv2.cvtColor(cv2.imread(resolve(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return cv2.resize(img, (SIZE, SIZE))


def hsv_mask(img, rule):
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (H >= rule["h"][0]) & (H <= rule["h"][1])
    if "h2" in rule: m |= (H >= rule["h2"][0]) & (H <= rule["h2"][1])
    return m & (S >= rule["s"][0]) & (S <= rule["s"][1]) & (V >= rule["v"][0]) & (V <= rule["v"][1])


def _band(x, lo, hi, w=12):
    return np.clip((x - lo) / w, 0, 1) * np.clip((hi - x) / w, 0, 1)


def our_tritan(img, deg=DEG, guided=True):
    """blue(H~90-135)->보라(H+), yellow(H~18-40)->주황(H-). S,V 유지. 경계 guided 평활."""
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat_g = np.clip((S - 18) / 50, 0, 1)                 # 회색 제외(hue밴드가 주 배제, floor 낮춤→pale_sky 커버)
    g_blue = sat_g * _band(H, 90, 135)                   # 파랑
    g_yellow = sat_g * _band(H, 18, 40)                  # 노랑
    Hn = (H + g_blue * deg + g_yellow * deg) % 180       # 파랑 H+ (보라), 노랑도 H+ (연두쪽) — 주황(-)은 CRR<1이라 반전
    hsv[..., 0] = Hn
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.
    if guided:
        delta = out - img
        r = max(2, SIZE // 16)
        delta = guided_filter(img, delta, r, 1e-3, max_side=2048)
        out = np.clip(img + delta, 0, 1)
    return out


rules = MAN["hsv_mask_rules"]
BLUE = MAN["role"]["blue_coverage"]; OFF = MAN["role"]["selectivity_off"]
per_img = []
for it in MAN["images"]:
    img = load(it["path"]); cat = it["category"]
    out = our_tritan(img)
    de_full = ciede2000(to_lab(img), to_lab(out))
    m = hsv_mask(img, rules[cat])
    de_mask = float(de_full[m].mean()) if m.sum() >= 50 else float("nan")
    per_img.append(dict(path=it["path"], cat=cat, npx=int(m.sum()),
                        dE_mask=round(de_mask, 2), dE_full=round(float(de_full.mean()), 2),
                        satD=round(float(sat(out).mean() - sat(img).mean()), 4),
                        crr=round(float(crr(img, out, "t")), 3)))

cats = {}
for r in per_img:
    cats.setdefault(r["cat"], []).append(r)
agg = {}
for c, rs in cats.items():
    dm = [r["dE_mask"] for r in rs if not np.isnan(r["dE_mask"])]
    agg[c] = dict(n=len(rs),
                  dE_mask_mean=round(float(np.mean(dm)), 2) if dm else None,
                  dE_mask_min=round(float(np.min(dm)), 2) if dm else None,
                  dE_full_mean=round(float(np.mean([r["dE_full"] for r in rs])), 2),
                  satD_mean=round(float(np.mean([r["satD"] for r in rs])), 4),
                  crr_mean=round(float(np.mean([r["crr"] for r in rs])), 3))

cover = BLUE + ["yellow"]
checks = {}
checks["coverage_dE_mask>=4"] = all(agg[c]["dE_mask_mean"] >= 4 for c in cover if c in agg)
checks["coverage_dE_min>=2"] = all(agg[c]["dE_mask_min"] >= 2 for c in cover if c in agg)
checks["distinguish_CRR>=1.0"] = all(agg[c]["crr_mean"] >= 1.0 for c in cover if c in agg)
checks["nowashout_|satD|<=0.03"] = all(abs(agg[c]["satD_mean"]) <= 0.03 for c in agg)
checks["selectivity_offdE<=1.5"] = all(agg[c]["dE_mask_mean"] <= 1.5 for c in OFF if c in agg)  # 마스크영역(대상색)만 — 혼합이미지 배경 confound 제거
blue_means = [agg[c]["dE_mask_mean"] for c in BLUE if c in agg]

result = dict(deg=DEG, per_img=per_img, agg=agg, checks=checks,
              blue_dE_std=round(float(np.std(blue_means)), 2), overall=all(checks.values()))
(OUT / "hue_method_scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

order = BLUE + ["yellow"] + OFF + ["skin"]
print(f"{'category':11} {'n':>2} | {'dE_mask':>7} {'min':>5} {'CRR':>5} {'satD':>7} {'dE_full':>7}")
for c in order:
    if c not in agg: continue
    a = agg[c]
    print(f"{c:11} {a['n']:>2} | {str(a['dE_mask_mean']):>7} {str(a['dE_mask_min']):>5} {a['crr_mean']:>5} {a['satD_mean']:>7} {a['dE_full_mean']:>7}")
print(f"\nblue dE std: {result['blue_dE_std']}")
print("\n=== 채점 (출력기반 지표) ===")
for k, v in checks.items():
    print(f"  {k:26} {'PASS' if v else 'FAIL'}")
print(f"\nOVERALL: {'PASS' if result['overall'] else 'FAIL'}")

# montage
reps = {}
for it in MAN["images"]:
    reps.setdefault(it["category"], it["path"])
cs = [c for c in order if c in reps]
fig, ax = plt.subplots(2, len(cs), figsize=(len(cs) * 1.7, 3.7))
for j, c in enumerate(cs):
    img = load(reps[c]); out = our_tritan(img)
    ax[0, j].imshow(img); ax[0, j].set_title(c, fontsize=7); ax[0, j].axis("off")
    ax[1, j].imshow(out); ax[1, j].set_title(f"dE {ciede2000(to_lab(img),to_lab(out)).mean():.1f}", fontsize=7); ax[1, j].axis("off")
fig.suptitle(f"our tritan (hue rot deg={DEG:g}, blue->violet / yellow->orange, guided): orig / out", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "hue_method_montage.png", dpi=95, bbox_inches="tight")
print("\n[saved] hue_method_scores.json, hue_method_montage.png")
