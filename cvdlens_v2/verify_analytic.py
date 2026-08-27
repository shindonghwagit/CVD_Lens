"""[선행] 배포 analytic HSV 회전(main._correct_image, t) 검증·확정.
원 기준(DECISION_TABLE.md L83) + 사용자 지정(red/green/gray≤0.1)로 test-set 21장 채점.
고정각도 확정을 위해 severity=1.0(=30°) 고정. w-gate가 아니라 배포된 hue-gate 버전(작동본).
산출물: reports/tritan_analytic_v2/analytic_scores.json.
"""
import sys, json
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "cvd-lens" / "inference"))
import main  # 배포 추론 (t = _tritan_hue_shift + guided)
from cvdlens_v2.artifact_probe import ciede2000, to_lab
from cvdlens_v2.daily_test import sat

SEV = 1.0
MAN = json.load(open("cvdlens_v2/testsets/tritan_blue/manifest.json")); VAL = MAN["coco_val_dir"]; rules = MAN["hsv_mask_rules"]
OUT = Path("reports/tritan_analytic_v2"); OUT.mkdir(parents=True, exist_ok=True)

def load(p):
    p = f"{VAL}/{p[5:]}" if p.startswith("coco:") else p
    return cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32)/255., (256,256))
def mask(img, r):
    h = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV); H,S,V = h[...,0],h[...,1],h[...,2]
    m = (H>=r["h"][0])&(H<=r["h"][1])
    if "h2" in r: m |= (H>=r["h2"][0])&(H<=r["h2"][1])
    return m&(S>=r["s"][0])&(S<=r["s"][1])&(V>=r["v"][0])&(V<=r["v"][1])

per=[]
for it in MAN["images"]:
    img=load(it["path"]); cat=it["category"]
    out=main._correct_image(img,"t",SEV)
    de=ciede2000(to_lab(img),to_lab(out)); m=mask(img,rules[cat])
    per.append(dict(path=it["path"],cat=cat,
        dE_mask=round(float(de[m].mean()),3) if m.sum()>=50 else None,
        satD=round(float(sat(out).mean()-sat(img).mean()),4)))
json.dump(per, open(OUT/"analytic_scores.json","w"), indent=1, ensure_ascii=False)

def cmean(c,k):
    vs=[r[k] for r in per if r["cat"]==c and r[k] is not None]; return round(float(np.mean(vs)),3) if vs else None
bs=[r for r in per if r["path"].endswith("blue_sea.jpg")][0]
print("이미지별:")
for r in per: print(f"  {r['cat']:11} {r['path'].split('/')[-1][:16]:16} dE_mask={r['dE_mask']} satD={r['satD']:+.4f}")
print(f"\n=== 원 기준 채점 (severity={SEV} 고정, 21장) ===")
print(f"[satΔ]   blue_sea satΔ = {bs['satD']:+.4f}  [≥-0.08?] {'PASS' if bs['satD']>=-0.08 else 'FAIL'}")
print(f"[커버]   blue_sea blueΔE = {bs['dE_mask']}  [≥5?] {'PASS' if bs['dE_mask']>=5 else 'FAIL'}")
print(f"[선택성] red_ctrl  dE_mask = {cmean('red_ctrl','dE_mask')}  [≤0.1?] {'PASS' if (cmean('red_ctrl','dE_mask') or 9)<=0.1 else 'FAIL'}")
print(f"[선택성] green_ctrl dE_mask = {cmean('green_ctrl','dE_mask')}  [≤0.1?] {'PASS' if (cmean('green_ctrl','dE_mask') or 9)<=0.1 else 'FAIL'}")
print(f"[선택성] gray_ctrl dE_mask = {cmean('gray_ctrl','dE_mask')}  [≤0.1?] {'PASS' if (cmean('gray_ctrl','dE_mask') or 9)<=0.1 else 'FAIL'}")
print(f"[참고]   skin dE_mask = {cmean('skin','dE_mask')}  | p/d: analytic은 t전용, p/d onnx 불변 → 회귀 없음")
