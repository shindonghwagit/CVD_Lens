"""Pareto 오버레이 — fresh 체크포인트(v2) + v3 + 2단계 파인튜닝 궤적.

x=blueΔE(blue_sea), y=satΔ. 수용영역(x≥5 & y≥−0.08) 음영. 2단계는 step0(=v2 step20000)에서
화살표로 궤적. 논문 figure 후보(§7 3중 음성).
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("reports/tritan_retrain_eval")
TH_BLUE, TH_SAT = 5.0, -0.08

# v2 fresh frontier (ckpt_scan.json)
sc = json.load(open(OUT / "ckpt_scan.json"))
v2 = [(r["blue_sea"]["blueDE"], r["blue_sea"]["satDE"], r["step"]) for r in sc]

# v3 fresh (L_sat from-scratch)
v3_pt = (2.73, 0.0041)   # eval_after_v3.json

# 2-stage trajectory: step0 = v2 step20000 endpoint, then finetune ckpts
tj = json.load(open(OUT / "stage2/stage2_trajectory.json"))
stage2 = [(7.78, -0.1651, 0)] + [(r["blue_sea"]["blueDE"], r["blue_sea"]["satDE"], r["step"]) for r in tj]

fig, ax = plt.subplots(figsize=(8.2, 5.6))
xmax = 9.2
# acceptance region
ax.fill_between([TH_BLUE, xmax], TH_SAT, 0.06, color="green", alpha=0.10, label="accept region (x>=5 & y>=-0.08)")

# v2 frontier
xs = [p[0] for p in v2]; ys = [p[1] for p in v2]
ax.plot(xs, ys, "-o", color="steelblue", label="fresh checkpoints (no L_sat)", zorder=3)
for x, y, st in v2:
    ax.annotate(f"{st//1000}k", (x, y), textcoords="offset points", xytext=(4, -10), fontsize=8, color="steelblue")

# v3 point
ax.scatter([v3_pt[0]], [v3_pt[1]], marker="X", s=140, color="crimson", zorder=5,
           label="v3 fresh + L_sat=200 (blue collapse)")
ax.annotate("v3", (v3_pt[0], v3_pt[1]), textcoords="offset points", xytext=(6, 4), fontsize=9, color="crimson")

# 2-stage trajectory with arrows
sx = [p[0] for p in stage2]; sy = [p[1] for p in stage2]
ax.plot(sx, sy, "--", color="darkorange", alpha=0.6, zorder=4)
for i in range(len(stage2) - 1):
    ax.annotate("", xy=(sx[i+1], sy[i+1]), xytext=(sx[i], sy[i]),
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.6), zorder=4)
ax.scatter(sx, sy, marker="s", s=45, color="darkorange", zorder=5,
           label="2-stage finetune (step20000 -> +L_sat=500)")
for x, y, st in stage2:
    ax.annotate(f"{st}", (x, y), textcoords="offset points", xytext=(4, 6), fontsize=8, color="darkorange")

ax.axvline(TH_BLUE, ls="--", color="gray", lw=1)
ax.axhline(TH_SAT, ls="--", color="crimson", lw=1)
ax.set_xlabel("blue_sea  blueDE00 (correction magnitude)")
ax.set_ylabel("blue_sea  satDelta (HSV)")
ax.set_title("Tritan operating-point search: fresh frontier / v3 / 2-stage trajectory\n"
             "(no point satisfies blueDE>=5 & satDelta>=-0.08 with gate+p/d)")
ax.legend(loc="lower left", fontsize=8)
ax.set_xlim(1.5, xmax); ax.set_ylim(-0.20, 0.06)
fig.tight_layout(); fig.savefig(OUT / "pareto_2stage_overlay.png", dpi=120)
print("[saved]", OUT / "pareto_2stage_overlay.png")
print("2-stage traj (step, blueDE, satD):", [(s, round(x,2), round(y,3)) for x,y,s in stage2])
