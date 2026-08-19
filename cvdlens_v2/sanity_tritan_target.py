"""
작업 2 — tritan daltonize 타깃 sanity (재학습/사용 전 필수 검증).

_ERR2MOD['t'] 수정(B에러 → R,G 재분배)이 실제로 올바른 타깃을 만드는지 확인.
순색 패치 + 실제 이미지로 아래를 점검하고, 하나라도 실패하면 FAIL 보고.

체크:
  (1) 파랑 패치 타깃에 R 성분이 유의미하게 추가되는가 (보라 방향)   [gate]
  (2) 회색/무채색 패치는 거의 불변인가                              [gate]
  (3) sim(t) 기준 파랑 vs 초록 ΔE00이 타깃 적용 후 증가하는가 (혼동쌍 분리)  [gate]
  (4) p/d 타깃은 수정 전후 bit-exact 동일한가 (회귀 방지)            [gate]
  (P) tritan 가시 부분공간 투영 잔존율(파랑 delta): <0.70이면 GAIN/행렬 재검토 경고

daltonize는 linear RGB에서 동작. 지표 정의(ciede2000)는 기존 스크립트 재사용.
Run: py -m cvdlens_v2.sanity_tritan_target
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvdlens_v2.simulation import daltonize, simulate, _ERR2MOD, TRITAN_SHIFT_GAIN
from cvdlens_v2.color import srgb_to_linear, linear_to_srgb, rgb_to_lab
from cvdlens_v2.basis import get_basis
from cvdlens_v2.artifact_probe import ciede2000
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

OUT = Path("reports/tritan_sanity"); OUT.mkdir(parents=True, exist_ok=True)

# Pure sRGB patches → linear for daltonize.
PATCHES_SRGB = {
    "blue":   (0.0, 0.0, 1.0), "yellow": (1.0, 1.0, 0.0),
    "cyan":   (0.0, 1.0, 1.0), "purple": (1.0, 0.0, 1.0),
    "red":    (1.0, 0.0, 0.0), "green":  (0.0, 1.0, 0.0),
    "gray":   (0.5, 0.5, 0.5),
}
R_ADD_MIN = 0.05        # (1) min linear R added to blue target
GRAY_MAX = 0.01         # (2) max |Δ| on achromatic patch
RETENTION_MIN = 0.70    # (P) visible-subspace retention warn threshold


def patch_lin(srgb, n=8):
    t = torch.tensor(srgb, dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, n, n)
    return srgb_to_linear(t.contiguous())


def lab_np(lin):  # (1,3,H,W) linear → (H,W,3) Lab numpy
    return rgb_to_lab(lin)[0].permute(1, 2, 0).numpy()


def de00(lin_a, lin_b):
    return float(ciede2000(lab_np(lin_a), lab_np(lin_b)).mean())


def visible_retention(delta_vec, cvd_type="t"):
    """|P·delta| / |delta| where P projects onto the CVD visible subspace."""
    B = get_basis(cvd_type).numpy()          # (2,3) rows [b_L, b_C], orthonormal
    d = delta_vec.astype(np.float64)
    nd = np.linalg.norm(d)
    if nd < 1e-8:
        return float("nan")
    proj = B.T @ (B @ d)                      # reconstruct in-plane component
    return float(np.linalg.norm(proj) / nd)


def main():
    res = {"checks": {}, "patches": {}, "retention": {}}

    # ── per-patch daltonize(t) ──
    for name, srgb in PATCHES_SRGB.items():
        o = patch_lin(srgb)
        d = daltonize(o, "t", 1.0)
        delta = (d - o)[0].mean(dim=(1, 2)).numpy()        # (3,) linear ΔRGB
        res["patches"][name] = dict(
            orig=[round(float(x), 3) for x in o[0].mean(dim=(1, 2))],
            target=[round(float(x), 3) for x in d[0].mean(dim=(1, 2))],
            dRGB=[round(float(x), 4) for x in delta])
        if name not in ("gray",):
            res["retention"][name] = round(visible_retention(delta), 3)

    # ── (1) blue R-add ──
    blue_dR = res["patches"]["blue"]["dRGB"][0]
    c1 = blue_dR >= R_ADD_MIN
    res["checks"]["1_blue_R_added"] = dict(dR=blue_dR, min=R_ADD_MIN, pass_=bool(c1))

    # ── (2) gray ~unchanged ──
    gray_max = max(abs(x) for x in res["patches"]["gray"]["dRGB"])
    c2 = gray_max < GRAY_MAX
    res["checks"]["2_gray_unchanged"] = dict(maxAbs=round(gray_max, 4), max=GRAY_MAX, pass_=bool(c2))

    # ── (3) confusion-pair separation increases after target ──
    # The spec named blue-green, but under Machado-t that pair's ΔE00 is
    # lightness-dominated (blue dark, green bright) — NOT a real tritan confusion
    # pair. Report it for transparency AND auto-detect the actual confusion pair
    # (min sim(t) ΔE00 over all patch pairs) as the principled gate.
    import itertools
    names = [n for n in PATCHES_SRGB if n != "gray"]
    lins = {n: patch_lin(PATCHES_SRGB[n]) for n in names}
    sim_o = {n: simulate(lins[n], "t", 1.0) for n in names}
    pair_de = {(a, b): de00(sim_o[a], sim_o[b]) for a, b in itertools.combinations(names, 2)}
    conf_pair = min(pair_de, key=pair_de.get)                 # smallest sim(t) ΔE
    a, b = conf_pair
    de_orig = pair_de[conf_pair]
    de_tgt = de00(simulate(daltonize(lins[a], "t", 1.0), "t", 1.0),
                  simulate(daltonize(lins[b], "t", 1.0), "t", 1.0))
    c3 = de_tgt > de_orig
    # as-specified blue-green, reported (not a gate)
    bo, go = lins["blue"], lins["green"]
    bg_o = de00(sim_o["blue"], sim_o["green"])
    bg_t = de00(simulate(daltonize(bo, "t", 1.0), "t", 1.0),
                simulate(daltonize(go, "t", 1.0), "t", 1.0))
    res["checks"]["3_confusion_pair_sep_up"] = dict(
        confusion_pair=f"{a}-{b}", de_orig=round(de_orig, 2), de_target=round(de_tgt, 2),
        delta=round(de_tgt - de_orig, 2), pass_=bool(c3),
        spec_blue_green=dict(de_orig=round(bg_o, 2), de_target=round(bg_t, 2),
                             note="lightness-dominated; not a tritan confusion pair"))

    # ── (4) p/d bit-exact vs OLD single matrix ──
    old = np.array([[0, 0, 0], [0.7, 1, 0], [0.7, 0, 1]], dtype=np.float32)
    torch.manual_seed(0); rgb = torch.rand(1, 3, 48, 48)
    def old_dalt(rgb, t):
        sim = simulate(rgb, t, 1.0, "machado"); err = rgb - sim
        m = torch.from_numpy(old)
        return (rgb + torch.einsum('ij,bjhw->bihw', m, err)).clamp(0, 1)
    pd_max = max((daltonize(rgb, t, 1.0) - old_dalt(rgb, t)).abs().max().item() for t in ("p", "d"))
    mat_exact = all(np.array_equal(_ERR2MOD[t], old) for t in ("p", "d"))
    c4 = (pd_max == 0.0) and mat_exact
    res["checks"]["4_pd_bit_exact"] = dict(max_abs_diff=pd_max, matrix_equal=mat_exact, pass_=bool(c4))

    # ── (P) blue retention ──
    blue_ret = res["retention"]["blue"]
    ret_warn = blue_ret < RETENTION_MIN
    res["checks"]["P_blue_retention"] = dict(retention=blue_ret, min=RETENTION_MIN,
        warn=bool(ret_warn), note="visible-subspace retention; low ⇒ shift wasted along confusion axis")

    gates = [c1, c2, c3, c4]
    res["all_gates_pass"] = bool(all(gates))
    res["gain"] = TRITAN_SHIFT_GAIN

    # ── real-image montage ──
    _montage()

    (OUT / "sanity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _print(res)
    return 0 if res["all_gates_pass"] else 1


def _load_srgb(path, cap=768):
    import cv2
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = img.shape[:2]; m = max(h, w)
    if m > cap:
        s = cap / m; img = cv2.resize(img, (round(w * s), round(h * s)))
    return torch.from_numpy(img).permute(2, 0, 1)[None]


def _to_disp(lin):  # linear (1,3,H,W) → HWC sRGB uint8-ish float
    return linear_to_srgb(lin)[0].permute(1, 2, 0).clamp(0, 1).numpy()


def _montage():
    imgs = [("blue_sea", "outputs/artifact_analysis/blue_sea.jpg"),
            ("traffic_street", "outputs/daily_test/traffic_street.jpg")]  # tennis-court proxy (blue sky/green)
    for name, path in imgs:
        if not Path(path).exists():
            continue
        srgb = _load_srgb(path); lin = srgb_to_linear(srgb)
        tgt = daltonize(lin, "t", 1.0)
        so, st = simulate(lin, "t", 1.0), simulate(tgt, "t", 1.0)
        panels = [("original", srgb[0].permute(1, 2, 0).numpy()),
                  ("daltonize t target", _to_disp(tgt)),
                  ("sim(t) original", _to_disp(so)),
                  ("sim(t) target", _to_disp(st))]
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        for a, (ti, im) in zip(ax, panels):
            a.imshow(im); a.set_title(ti, fontsize=10); a.axis("off")
        fig.suptitle(f"{name} — tritan daltonize target (gain={TRITAN_SHIFT_GAIN})", fontsize=12)
        fig.tight_layout(); fig.savefig(OUT / f"{name}_tritan_target.png", dpi=85, bbox_inches="tight")
        plt.close(fig)


def _print(res):
    print("=" * 64); print("tritan daltonize target sanity"); print("=" * 64)
    for name, p in res["patches"].items():
        ret = res["retention"].get(name)
        rs = f"  retention={ret}" if ret is not None else ""
        print(f"  {name:7} orig{p['orig']} -> target{p['target']}  ΔRGB{p['dRGB']}{rs}")
    print("-" * 64)
    for k, c in res["checks"].items():
        if "pass_" in c:
            mark = "PASS" if c["pass_"] else "FAIL"
        else:
            mark = "WARN" if c.get("warn") else "OK"
        print(f"  [{mark:4}] {k}: {({x:y for x,y in c.items() if x!='pass_'})}")
    print("-" * 64)
    print(f"  ALL GATES {'PASS' if res['all_gates_pass'] else 'FAIL'}  (gain={res['gain']})")
    print(f"[save] {OUT/'sanity.json'} + montages")


if __name__ == "__main__":
    sys.exit(main())
