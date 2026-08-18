"""
작업 1 배포 후 smoke test — 실제 Render 엔드포인트.

1) /health 200 대기(콜드스타트 포함).
2) 새 코드(guided ON) live 감지: 배포 응답을 로컬 guided-ON / guided-OFF 보정본과 비교,
   더 가까운 쪽으로 판별. guided-ON에 수렴할 때까지 폴링(재빌드 대기).
3) stop_sign, traffic_street 각 p/d 요청 → 200 + 유효 이미지 + guided-ON 확인.

env 토글(on/off)은 Render 대시보드 env 변경 권한이 없어 live 서버에서 직접 못 켜고/끈다.
→ 토글 코드 경로는 로컬에서 검증(main.py reload). live는 '기본 ON'만 출력 비교로 확인.

Run: py -m cvdlens_v2.smoke_deploy
"""
from __future__ import annotations
import sys, io, time, json, urllib.request, ssl
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvdlens_v2 import infer_local as il
from cvdlens_v2.daily_test import call_api, API
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

OUT = Path("reports/deploy_smoke"); OUT.mkdir(parents=True, exist_ok=True)
_CTX = ssl.create_default_context()
CASES = [("stop_sign", "outputs/artifact_analysis/stop_sign.jpg", ["p", "d"]),
         ("traffic_street", "outputs/daily_test/traffic_street.jpg", ["p", "d"])]
SEV = 1.0
POLL_TIMEOUT_S = 900
POLL_EVERY_S = 30


def health():
    try:
        urllib.request.urlopen(API + "/health", timeout=90, context=_CTX).read()
        return True
    except Exception as e:
        print(f"  [health] {repr(e)[:70]}"); return False


def classify(img, deployed_u8, t):
    """Return (label, d_on, d_off): which local variant the deployed output matches."""
    on = il.correct(img, t, SEV, use_guided=True)["corrected"]
    off = il.correct(img, t, SEV, use_guided=False)["corrected"]
    dep = deployed_u8.astype(np.float32) / 255.0
    if dep.shape != on.shape:
        import cv2
        dep = cv2.resize(dep, (on.shape[1], on.shape[0]))
    d_on = float(np.abs(dep - on).mean())
    d_off = float(np.abs(dep - off).mean())
    return ("guided_ON" if d_on < d_off else "guided_OFF"), d_on, d_off


def wait_for_deploy(probe_img, probe_t):
    print(f"[poll] waiting for guided-ON deploy (timeout {POLL_TIMEOUT_S}s)...")
    t0 = time.time()
    while time.time() - t0 < POLL_TIMEOUT_S:
        if health():
            try:
                dep = call_api((probe_img * 255 + 0.5).astype(np.uint8), probe_t, SEV, retries=2)
                label, d_on, d_off = classify(probe_img, dep, probe_t)
                el = int(time.time() - t0)
                print(f"  [{el}s] probe {probe_t}: {label}  (d_on={d_on:.4f} d_off={d_off:.4f})")
                if label == "guided_ON":
                    return True
            except Exception as e:
                print(f"  [probe] {repr(e)[:70]}")
        time.sleep(POLL_EVERY_S)
    return False


def main():
    probe_img = il.load_rgb(CASES[1][1])         # traffic_street (clear guided delta)
    if not wait_for_deploy(probe_img, "p"):
        print("\n[FAIL] guided-ON deploy not detected within timeout. "
              "Render 빌드 지연/실패 가능 — 대시보드 확인 필요.")
        return 1

    print("\n[smoke] 4 cases: 200 + valid image + guided-ON")
    rows, ok = [], True
    for name, path, types in CASES:
        img = il.load_rgb(path)
        for t in types:
            try:
                dep = call_api((img * 255 + 0.5).astype(np.uint8), t, SEV, retries=3)
                status200 = dep is not None and dep.ndim == 3 and dep.shape[2] == 3
                label, d_on, d_off = classify(img, dep, t)
                Image.fromarray(dep).save(OUT / f"{name}_{t}_deployed.jpg", quality=95)
                rows.append(dict(case=f"{name}_{t}", ok_image=bool(status200), guided=label,
                                 d_on=round(d_on, 4), d_off=round(d_off, 4)))
                ok = ok and status200 and label == "guided_ON"
                print(f"  {name}_{t}: 200={status200}  {label}  (d_on={d_on:.4f} d_off={d_off:.4f})")
            except Exception as e:
                ok = False; rows.append(dict(case=f"{name}_{t}", error=repr(e)[:80]))
                print(f"  {name}_{t}: ERROR {repr(e)[:80]}")

    (OUT / "smoke.json").write_text(json.dumps(dict(endpoint=API, all_pass=ok, rows=rows), indent=2),
                                    encoding="utf-8")
    print(f"\n[{'PASS' if ok else 'FAIL'}] deploy smoke — {API}")
    print(f"[save] {OUT/'smoke.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
