"""
OOM-fix 배포 확인 — 대용량 이미지 버스트 stress test.

native-guided(옛, OOM)와 512-guided(새, 수정) 출력은 JPEG q94 노이즈보다 작게 달라
출력 비교로는 구분 불가. 유일한 신뢰 신호 = 운영 안정성: 새 코드(2048² RSS ~225MB)는
대용량 요청 버스트를 견디고, 옛 코드(native 코어 ~577MB)는 버스트 중 OOM(스크린샷).

전략: /health 대기 → 대용량(stop_sign, food_tomatoes) BURST개 연속 요청. 전부 200+유효면
'clean burst' = 새 코드 안정. 실패(OOM→502/timeout) 있으면 배포 진행 중/옛 코드로 보고
대기 후 버스트 재시도. clean burst 또는 timeout까지.

Run: py -m cvdlens_v2.stress_deploy
"""
from __future__ import annotations
import sys, io, time, json, urllib.request, ssl
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvdlens_v2 import infer_local as il
from cvdlens_v2.daily_test import API
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

OUT = Path("reports/deploy_smoke"); OUT.mkdir(parents=True, exist_ok=True)
_CTX = ssl.create_default_context()
# heaviest requests: big saturated-red images (worst case for the guided filter)
LARGE = [("stop_sign", "outputs/artifact_analysis/stop_sign.jpg"),
         ("food_tomatoes", "outputs/daily_test/food_tomatoes.jpg")]
BURST = 10                      # old code OOM'd within ~4; 10 large ⇒ decisive
TOTAL_TIMEOUT_S = 900
BURST_GAP_S = 45


def _multipart(u8, cvd_type, severity):
    buf = io.BytesIO(); Image.fromarray(u8).save(buf, format="JPEG", quality=92)
    jpg = buf.getvalue(); b = b"----s"; CRLF = b"\r\n"; body = b""
    body += b"--"+b+CRLF+b'Content-Disposition: form-data; name="image"; filename="f.jpg"'+CRLF
    body += b"Content-Type: image/jpeg"+CRLF+CRLF+jpg+CRLF
    for k, v in [("cvd_type", cvd_type), ("severity", str(severity))]:
        body += b"--"+b+CRLF+f'Content-Disposition: form-data; name="{k}"'.encode()+CRLF+CRLF+v.encode()+CRLF
    body += b"--"+b+b"--"+CRLF
    return body, b


def request_once(u8, cvd_type, severity=1.0, timeout=120):
    body, b = _multipart(u8, cvd_type, severity)
    req = urllib.request.Request(API + "/infer", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={b.decode()}"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=timeout, context=_CTX).read()
    img = np.asarray(Image.open(io.BytesIO(r)).convert("RGB"))
    return img.ndim == 3 and img.shape[2] == 3, len(r), time.time() - t0


def health():
    try:
        urllib.request.urlopen(API + "/health", timeout=90, context=_CTX).read(); return True
    except Exception:
        return False


def run_burst(imgs):
    results = []
    for i in range(BURST):
        name, u8, t = imgs[i % len(imgs)]
        try:
            ok, nbytes, dt = request_once(u8, t)
            results.append(dict(i=i, case=f"{name}_{t}", ok=ok, kb=round(nbytes/1024), s=round(dt, 1)))
            print(f"    [{i+1}/{BURST}] {name}_{t}: 200 ok={ok} {nbytes//1024}KB {dt:.1f}s")
        except Exception as e:
            results.append(dict(i=i, case=f"{name}_{t}", ok=False, error=repr(e)[:80]))
            print(f"    [{i+1}/{BURST}] {name}_{t}: FAIL {repr(e)[:80]}")
    return results


def main():
    imgs = [(name, (il.load_rgb(p) * 255 + 0.5).astype(np.uint8), t)
            for (name, p) in LARGE for t in ("p", "d")]
    print(f"[stress] {BURST}-request large-image bursts until clean (OOM-fix confirm). "
          f"endpoint {API}")
    t0 = time.time(); attempt = 0; last = None
    while time.time() - t0 < TOTAL_TIMEOUT_S:
        attempt += 1
        if not health():
            print(f"  [{int(time.time()-t0)}s] health down (cold start / rebuild)…"); time.sleep(BURST_GAP_S); continue
        print(f"\n[burst {attempt}] ({int(time.time()-t0)}s elapsed)")
        res = run_burst(imgs); last = res
        n_ok = sum(1 for r in res if r.get("ok"))
        if n_ok == BURST:
            print(f"\n[PASS] clean burst — {BURST}/{BURST} large requests OK ⇒ OOM fix live & stable.")
            (OUT / "stress.json").write_text(json.dumps(dict(endpoint=API, clean=True,
                attempt=attempt, results=res), indent=2), encoding="utf-8")
            return 0
        print(f"  burst {n_ok}/{BURST} ok — deploy in progress or old code OOMing; retry in {BURST_GAP_S}s")
        time.sleep(BURST_GAP_S)
    print("\n[FAIL] no clean burst within timeout — check Render dashboard.")
    (OUT / "stress.json").write_text(json.dumps(dict(endpoint=API, clean=False, results=last), indent=2),
                                     encoding="utf-8")
    return 1


if __name__ == "__main__":
    sys.exit(main())
