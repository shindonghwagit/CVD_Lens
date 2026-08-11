"""
Daily-image correction QA through the DEPLOYED backend (core subset).

4 categories × 3 types × 2 severities (0.6, 1.0) = 24 cases, each hitting the
real Render/FastAPI endpoint the frontend uses. Saves original / corrected /
Brettel-sim(original) / Brettel-sim(corrected) + a montage, and quantitative
proxies (ΔE00, saturation change, confusion-weighted contrast ratio CRR).

Preprocessing matches the frontend: cap long side to 2048, send JPEG q92.
daltonize NOT imported. Pure test — no code under test is modified.

Run: py -m cvdlens_v2.daily_test
"""
from __future__ import annotations
import sys, io, json, time, urllib.request, ssl
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.color import srgb_to_linear, rgb_to_lab
from cvdlens_v2.confusion import compute_confusion_weight
from cvdlens_v2.simulation import simulate
from cvdlens_v2.artifact_probe import ciede2000, to_lab
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

API = "https://cvd-lens.onrender.com"
OUT = Path("outputs/daily_test")
CATS = ["food_tomatoes", "skin_portrait", "traffic_street", "nature_autumn"]
TYPES = ["p", "d", "t"]
SEVS = [0.6, 1.0]
MAX_SIDE = 2048
_CTX = ssl.create_default_context()


def cap2048(img_f32):
    h, w = img_f32.shape[:2]; m = max(h, w)
    if m > MAX_SIDE:
        s = MAX_SIDE / m
        img_f32 = cv2.resize(img_f32, (round(w*s), round(h*s)), interpolation=cv2.INTER_AREA)
    return img_f32


def _multipart(img_u8, cvd_type, severity):
    buf = io.BytesIO(); Image.fromarray(img_u8).save(buf, format="JPEG", quality=92)
    jpg = buf.getvalue()
    b = b"----cvdlens"; CRLF = b"\r\n"; body = b""
    body += b"--"+b+CRLF+b'Content-Disposition: form-data; name="image"; filename="f.jpg"'+CRLF
    body += b"Content-Type: image/jpeg"+CRLF+CRLF+jpg+CRLF
    for k, v in [("cvd_type", cvd_type), ("severity", str(severity))]:
        body += b"--"+b+CRLF+f'Content-Disposition: form-data; name="{k}"'.encode()+CRLF+CRLF+v.encode()+CRLF
    body += b"--"+b+b"--"+CRLF
    return body, b


def call_api(img_u8, cvd_type, severity, retries=4):
    body, b = _multipart(img_u8, cvd_type, severity)
    req = urllib.request.Request(API+"/infer", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={b.decode()}"})
    last = None
    for i in range(retries):
        try:
            r = urllib.request.urlopen(req, timeout=120, context=_CTX).read()
            return np.asarray(Image.open(io.BytesIO(r)).convert("RGB"))
        except Exception as e:
            last = e; print(f"    retry {i+1} ({repr(e)[:60]})"); time.sleep(5*(i+1))
    raise RuntimeError(f"API failed after {retries}: {last}")


def warmup():
    try:
        urllib.request.urlopen(API+"/health", timeout=90, context=_CTX).read()
        print("[warmup] backend healthy")
    except Exception as e:
        print(f"[warmup] {repr(e)[:80]}")


def sat(img):  # HSV saturation, (H,W,3) in [0,1]
    mx = img.max(2); mn = img.min(2); return (mx-mn)/(mx+1e-6)


def brettel_sim(img_f32, t):
    lin = srgb_to_linear(torch.from_numpy(img_f32).permute(2,0,1)[None].float())
    s = simulate(lin, t, 1.0, "brettel")
    # linear→srgb for display
    from cvdlens_v2.color import linear_to_srgb
    return linear_to_srgb(s)[0].permute(1,2,0).clamp(0,1).numpy()


def crr(orig_f32, corr_f32, t):
    """Confusion-weighted contrast ratio on the Brettel CVD view (>1 = recovered)."""
    o = srgb_to_linear(torch.from_numpy(orig_f32).permute(2,0,1)[None].float())
    c = srgb_to_linear(torch.from_numpy(corr_f32).permute(2,0,1)[None].float())
    w = compute_confusion_weight(o, t, 1.0)
    so, sc = simulate(o, t, 1.0, "brettel"), simulate(c, t, 1.0, "brettel")
    def cmag(x):
        lab = rgb_to_lab(x)
        dy = (lab[:,:,1:]-lab[:,:,:-1]); dx = (lab[:,:,:,1:]-lab[:,:,:,:-1])
        g = torch.zeros_like(lab[:,:1]); g[:,:,1:]+=(dy**2).sum(1,keepdim=True); g[:,:,:,1:]+=(dx**2).sum(1,keepdim=True)
        return torch.sqrt(g+1e-6)
    ws = w.sum()+1e-6
    return float(((cmag(sc)*w).sum()/ws)/(((cmag(so)*w).sum()/ws)+1e-6))


def main():
    OUT.mkdir(exist_ok=True); warmup()
    rows = []
    for cat in CATS:
        src = OUT / f"{cat}.jpg"
        if not src.exists(): print(f"[skip] {cat} missing"); continue
        orig = cap2048(cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
        orig_u8 = (orig*255+0.5).astype(np.uint8)
        for t in TYPES:
            simo = brettel_sim(orig, t)
            for sv in SEVS:
                tag = f"{cat}_{t}_s{sv}"
                d = OUT / tag; d.mkdir(exist_ok=True)
                try:
                    corr_u8 = call_api(orig_u8, t, sv)
                except Exception as e:
                    print(f"[FAIL] {tag}: {e}"); continue
                corr = corr_u8.astype(np.float32)/255.0
                if corr.shape != orig.shape:
                    corr = cv2.resize(corr, (orig.shape[1], orig.shape[0]))
                simc = brettel_sim(corr, t)
                # save
                for nm, arr in [("original", orig), ("corrected", corr), ("sim_orig", simo), ("sim_corr", simc)]:
                    cv2.imwrite(str(d/f"{nm}.jpg"), cv2.cvtColor((arr*255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                de = ciede2000(to_lab(orig), to_lab(corr))
                rec = dict(tag=tag, cat=cat, type=t, sev=sv,
                           deE_mean=round(float(de.mean()),2), deE_p99=round(float(np.percentile(de,99)),2),
                           sat_delta=round(float(sat(corr).mean()-sat(orig).mean()),4),
                           crr=round(crr(orig, corr, t),3))
                rows.append(rec)
                # montage
                fig, ax = plt.subplots(1, 4, figsize=(18, 5))
                for a,(ti,im) in zip(ax, [("original",orig),(f"corrected {t} s{sv}",corr),
                                          (f"sim({t}) orig",simo),(f"sim({t}) corr",simc)]):
                    a.imshow(im); a.set_title(ti, fontsize=9); a.axis("off")
                fig.suptitle(f"{tag}  ΔE00={rec['deE_mean']}  Δsat={rec['sat_delta']:+.3f}  CRR={rec['crr']}", fontsize=11)
                fig.tight_layout(); fig.savefig(d/"montage.png", dpi=85, bbox_inches="tight"); plt.close(fig)
                print(f"[ok] {tag}  ΔE={rec['deE_mean']}  Δsat={rec['sat_delta']:+.3f}  CRR={rec['crr']}")
    (OUT/"daily_stats.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n[save] {OUT/'daily_stats.json'} ({len(rows)} cases)")


if __name__ == "__main__":
    main()
