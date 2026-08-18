"""
작업 2 — API 응답 JPEG quality 스윕.

배경: guided ON에서도 CRR 경계 케이스(traffic p/d, food d)가 API 응답 JPEG q92 재압축으로
1 아래로 떨어진다. quality를 올리면(또는 PNG 무손실) 회복 마진이 보존되는지, 파일크기
대가는 얼마인지 측정한다.

대상: traffic_street(p/d), food_tomatoes(d) @ sev 1.0, guided ON.
quality ∈ {92,94,95,96,98} + PNG(무손실 기준점). raw(비압축)도 상한 기준으로 병기.
각 quality: CRR(기존 daily_test.crr 정의) + 응답 바이트 크기(원해상도 인코딩).

선정 기준: 경계 3케이스 모두 CRR ≥ 1.0 을 만족하는 최소 JPEG quality.
q98에서도 <1 케이스가 남으면 config 준비만 하고 push 전 보고.

Run: py -m cvdlens_v2.jpeg_sweep
"""
from __future__ import annotations
import sys, io, json
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvdlens_v2 import infer_local as il
from cvdlens_v2.daily_test import crr
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

OUT = Path("reports/jpeg_sweep"); OUT.mkdir(parents=True, exist_ok=True)
TARGETS = [("traffic_street", "p"), ("traffic_street", "d"), ("food_tomatoes", "d")]
QUALITIES = [92, 94, 95, 96, 98]
SEV = 1.0


def encode_decode(u8, fmt, quality=None):
    buf = io.BytesIO()
    if fmt == "JPEG":
        Image.fromarray(u8).save(buf, format="JPEG", quality=quality)
    else:
        Image.fromarray(u8).save(buf, format="PNG")
    data = buf.getvalue()
    dec = np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(np.float32) / 255.0
    return dec, len(data)


def main():
    rows = []           # (cat,type,fmt,quality,crr,bytes)
    per_case = {}
    for cat, t in TARGETS:
        img = il.load_rgb(f"outputs/daily_test/{cat}.jpg")
        r = il.correct(img, t, SEV, use_guided=True)
        corr = r["corrected"]
        u8 = (corr * 255 + 0.5).astype(np.uint8)
        key = f"{cat}_{t}"
        per_case[key] = {}

        crr_raw = crr(img, corr, t)                       # uncompressed upper bound
        rows.append(dict(case=key, fmt="raw", quality=None, crr=round(crr_raw, 3), bytes=None))
        per_case[key]["raw"] = round(crr_raw, 3)

        dec_png, nbytes = encode_decode(u8, "PNG")
        crr_png = crr(img, dec_png, t)
        rows.append(dict(case=key, fmt="PNG", quality=None, crr=round(crr_png, 3), bytes=nbytes))
        per_case[key]["PNG"] = round(crr_png, 3)

        for q in QUALITIES:
            dec, nbytes = encode_decode(u8, "JPEG", q)
            c = crr(img, dec, t)
            rows.append(dict(case=key, fmt="JPEG", quality=q, crr=round(c, 3), bytes=nbytes))
            per_case[key][f"q{q}"] = round(c, 3)
        print(f"[{key}] raw={crr_raw:.3f} png={crr_png:.3f} "
              + " ".join(f"q{q}={per_case[key][f'q{q}']:.3f}" for q in QUALITIES))

    # ── selection: min JPEG quality where ALL boundary cases CRR >= 1.0 ──
    selected = None
    for q in QUALITIES:
        if all(per_case[f"{cat}_{t}"][f"q{q}"] >= 1.0 for cat, t in TARGETS):
            selected = q; break
    png_ok = all(per_case[f"{cat}_{t}"]["PNG"] >= 1.0 for cat, t in TARGETS)

    (OUT / "jpeg_sweep.json").write_text(
        json.dumps(dict(rows=rows, per_case=per_case, selected_quality=selected,
                        png_all_recovered=png_ok), indent=2), encoding="utf-8")
    _write_md(rows, per_case, selected, png_ok)
    verdict = (f"selected JPEG quality = {selected}" if selected
               else f"NO JPEG quality in {QUALITIES} recovers all; PNG_all>=1.0={png_ok}")
    print(f"\n[verdict] {verdict}")
    print(f"[save] {OUT/'jpeg_sweep.json'} + REPORT.md")
    return selected, png_ok


def _write_md(rows, per_case, selected, png_ok):
    cases = list(per_case.keys())
    L = ["# API 응답 JPEG quality 스윕 (작업 2)", "",
         "guided ON, sev 1.0. CRR은 기존 daily_test.crr 정의(sim severity 1.0). "
         "각 quality로 원해상도 인코딩→디코딩 후 CRR 측정. raw=무압축 상한.", "",
         "## quality vs CRR (경계 케이스 3개, ≥1.0 굵게 판정)", "",
         "| fmt/quality | " + " | ".join(cases) + " |",
         "|---|" + "|".join("---" for _ in cases) + "|"]
    def fmt_crr(v): return f"**{v}**" if v is not None and v >= 1.0 else f"{v}"
    for label, key in [("raw", "raw"), ("PNG", "PNG")] + [(f"JPEG q{q}", f"q{q}") for q in QUALITIES]:
        L.append(f"| {label} | " + " | ".join(fmt_crr(per_case[c].get(key)) for c in cases) + " |")

    L += ["", "## quality vs 응답 파일크기 (bytes, 원해상도)", "",
          "| fmt/quality | " + " | ".join(cases) + " |",
          "|---|" + "|".join("---" for _ in cases) + "|"]
    bytes_by = {}
    for r in rows:
        if r["bytes"] is not None:
            lab = "PNG" if r["fmt"] == "PNG" else f"q{r['quality']}"
            bytes_by.setdefault(lab, {})[r["case"]] = r["bytes"]
    for lab in ["PNG"] + [f"q{q}" for q in QUALITIES]:
        row = bytes_by.get(lab, {})
        L.append(f"| {lab} | " + " | ".join(f"{row.get(c,0)/1024:.0f} KB" for c in cases) + " |")

    L += ["", "## 선정", ""]
    if selected:
        # size premium of selected q vs q92 baseline
        prem = []
        for c in cases:
            b92 = bytes_by.get("q92", {}).get(c, 0); bq = bytes_by.get(f"q{selected}", {}).get(c, 0)
            if b92: prem.append((bq / b92 - 1) * 100)
        avg_prem = sum(prem) / len(prem) if prem else 0
        L.append(f"**선정 JPEG quality = {selected}** — 경계 3케이스 모두 CRR ≥ 1.0 을 만족하는 최소값. "
                 f"q92 대비 응답 크기 평균 +{avg_prem:.0f}%. config `RESPONSE_JPEG_QUALITY` 기본값으로 반영.")
    else:
        L.append(f"**선정 실패** — {QUALITIES} 어느 quality도 3케이스 전부 회복 못 함 "
                 f"(PNG 무손실 전부 ≥1.0 = {png_ok}). config 준비만 하고 push 전 보고.")
    L.append("")
    (OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
