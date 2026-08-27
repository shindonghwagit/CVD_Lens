"""[부록] 학습-각도 HSV 회전 1회 스모크 (자체완결형 — 의존파일 삭제 대비).
theta-net(learned_tritan_train.py)과 동일 구조·손실·하이퍼파라미터·스텝·seed. 회전 함수만
YCbCr→미분가능 HSV(H만 회전, S·V 고정)로 교체. 나머지 전부 통제. 스윕·커리큘럼·재시도 금지.
결론(analytic 채택)에 영향 없음 — PASS/FAIL 이분법 채점만.
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvdlens_v2.color import srgb_to_linear, rgb_to_lab
from cvdlens_v2.simulation import simulate
from cvdlens_v2.train import COCOCropDataset
from cvdlens_v2.daily_test import sat, crr
from cvdlens_v2.artifact_probe import ciede2000, to_lab

MAX_ANGLE = 40.0 * np.pi / 180.0     # theta-net과 동일

# ── 회전: 미분가능 HSV (H만 회전, S·V 고정) — YCbCr 대비 유일 변경점 ──
def rgb2hsv(rgb):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    maxc = rgb.max(1, keepdim=True).values; minc = rgb.min(1, keepdim=True).values
    v = maxc; delta = maxc - minc; s = delta / (maxc + 1e-8); dp = delta + 1e-8
    rc, gc, bc = (maxc - r)/dp, (maxc - g)/dp, (maxc - b)/dp
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    return (h/6.0) % 1.0, s, v

def hsv2rgb(h, s, v):
    h6 = (h*6.0) % 6.0; c = v*s; x = c*(1 - torch.abs((h6 % 2.0) - 1)); z = torch.zeros_like(h); mm = v - c
    c0,c1,c2,c3,c4 = h6<1,(h6>=1)&(h6<2),(h6>=2)&(h6<3),(h6>=3)&(h6<4),(h6>=4)&(h6<5)
    r = torch.where(c0,c,torch.where(c1,x,torch.where(c2,z,torch.where(c3,z,torch.where(c4,x,c)))))
    g = torch.where(c0,x,torch.where(c1,c,torch.where(c2,c,torch.where(c3,x,torch.where(c4,z,z)))))
    b = torch.where(c0,z,torch.where(c1,z,torch.where(c2,x,torch.where(c3,c,torch.where(c4,c,x)))))
    return torch.cat([r+mm, g+mm, b+mm], 1)

def rotate_hue_raw(srgb, theta):
    h, s, v = rgb2hsv(srgb.clamp(0,1))
    return hsv2rgb((h + theta/(2*math.pi)) % 1.0, s, v)   # HSV2RGB 항상 in-gamut → clip 없음

def rotate_hue(srgb, theta):
    return rotate_hue_raw(srgb, theta).clamp(0, 1)

# ── 이하 theta-net과 동일(복붙) ──
class ThetaNet(nn.Module):
    def __init__(self, ch=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,ch,3,padding=1), nn.ReLU(),
            nn.Conv2d(ch,ch,3,padding=1,dilation=1), nn.ReLU(),
            nn.Conv2d(ch,ch,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(ch,ch,3,padding=4,dilation=4), nn.ReLU(),
            nn.Conv2d(ch,1,1))
    def forward(self, srgb): return torch.tanh(self.net(srgb)) * MAX_ANGLE

def _band(x, lo, hi, w=12.0):
    return np.clip((x-lo)/w,0,1) * np.clip((hi-x)/w,0,1)

def hue_gate(o_srgb):
    arr = (o_srgb.detach().clamp(0,1).numpy()*255).astype(np.uint8); gs=[]
    for im in arr:
        hsv = cv2.cvtColor(im.transpose(1,2,0), cv2.COLOR_RGB2HSV).astype(np.float32)
        H,S = hsv[...,0],hsv[...,1]; sat_g = np.clip((S-40)/60,0,1)
        gs.append(sat_g * np.maximum(_band(H,90,135), _band(H,18,40)))
    return torch.from_numpy(np.stack(gs))[:,None].float()

def vis_loss(o_srgb, theta, lam_sel=25.0, lam_sat=22.0, lam_reg=1.0):
    o_lin = srgb_to_linear(o_srgb); lab_o_sim = rgb_to_lab(simulate(o_lin,"t",1.0)); w = hue_gate(o_srgb)
    out_raw = rotate_hue_raw(o_srgb, theta); out = out_raw.clamp(0,1)
    lab_c_sim = rgb_to_lab(simulate(srgb_to_linear(out),"t",1.0))
    dE = torch.sqrt(((lab_c_sim - lab_o_sim)**2).sum(1,keepdim=True)+1e-6)
    L_vis = -(w*dE).mean(); L_sel = ((1-w)*theta**2).mean()
    L_gamut = (torch.relu(out_raw-1)+torch.relu(-out_raw)).mean(); L_reg = (theta**2).mean()
    return L_vis + lam_sel*L_sel + lam_sat*L_gamut + lam_reg*L_reg, float(L_vis), float(L_sel), float(L_gamut)

def train(steps=800, batch=8, crop=128, lr=2e-3):
    net = ThetaNet(); opt = torch.optim.Adam(net.parameters(), lr=lr)
    ds = COCOCropDataset("C:/Users/SCH/coco/val2017", crop=crop)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    it = iter(loader); net.train(); print(f"[train-HSV] steps={steps} batch={batch} crop={crop}")
    for s in range(1, steps+1):
        try: b = next(it)
        except StopIteration: it = iter(loader); b = next(it)
        loss, lv, lsel, lg = vis_loss(b, net(b))
        opt.zero_grad(); loss.backward(); opt.step()
        if s%100==0 or s==1: print(f"  step {s:>4}  L_vis={lv:.3f} L_sel={lsel:.4f} L_gamut={lg:.4f}")
    return net.eval()

def evaluate(net):
    MAN = json.load(open("cvdlens_v2/testsets/tritan_blue/manifest.json")); VAL=MAN["coco_val_dir"]; rules=MAN["hsv_mask_rules"]
    def load(p):
        p = f"{VAL}/{p[5:]}" if p.startswith("coco:") else p
        return cv2.resize(cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB).astype(np.float32)/255.,(256,256))
    def mask(img,r):
        h=cv2.cvtColor((img*255).astype(np.uint8),cv2.COLOR_RGB2HSV); H,S,V=h[...,0],h[...,1],h[...,2]
        mm=(H>=r["h"][0])&(H<=r["h"][1])
        if "h2" in r: mm|=(H>=r["h2"][0])&(H<=r["h2"][1])
        return mm&(S>=r["s"][0])&(S<=r["s"][1])&(V>=r["v"][0])&(V<=r["v"][1])
    per=[]
    for it in MAN["images"]:
        img=load(it["path"]); cat=it["category"]
        with torch.no_grad():
            t=torch.from_numpy(img).permute(2,0,1)[None].float(); out=rotate_hue(t,net(t))[0].permute(1,2,0).numpy()
        de=ciede2000(to_lab(img),to_lab(out)); mm=mask(img,rules[cat])
        per.append(dict(path=it["path"],cat=cat,
            dE_mask=round(float(de[mm].mean()),3) if mm.sum()>=50 else None,
            satD=round(float(sat(out).mean()-sat(img).mean()),4)))
    json.dump(per, open("reports/tritan_analytic_v2/hsv_scores.json","w"), indent=1, ensure_ascii=False)
    def cmean(c,k):
        vs=[r[k] for r in per if r["cat"]==c and r[k] is not None]; return round(float(np.mean(vs)),3) if vs else None
    bs=[r for r in per if r["path"].endswith("blue_sea.jpg")][0]
    print(f"\n=== 원 기준 채점 (HSV 학습-각도, 21장) — 파일: hsv_scores.json ===")
    print(f"[satΔ]   blue_sea satΔ  = {bs['satD']:+.4f}  [≥-0.08?] {'PASS' if bs['satD']>=-0.08 else 'FAIL'}")
    print(f"[커버]   blue_sea blueΔE= {bs['dE_mask']}  [≥5?] {'PASS' if bs['dE_mask']>=5 else 'FAIL'}")
    for c in ("red_ctrl","green_ctrl","gray_ctrl"):
        v=cmean(c,'dE_mask'); print(f"[선택성] {c:10} dE_mask = {v}  [≤0.1?] {'PASS' if (v or 9)<=0.1 else 'FAIL'}")
    print(f"[참고]   skin dE_mask = {cmean('skin','dE_mask')} | teal satΔ = {cmean('teal','satD')} | 물빠짐 max|satD| = {max(abs(r['satD']) for r in per):.4f}")

if __name__ == "__main__":
    torch.manual_seed(0)
    t = torch.rand(2,3,16,16)
    print(f"[sanity] HSV roundtrip(θ=0) max|Δ|={(rotate_hue(t,torch.zeros(2,1,16,16))-t).abs().max().item():.2e}")
    net = train(); evaluate(net)
    torch.save(net.state_dict(), "reports/tritan_analytic_v2/theta_net_hsv.pt"); print("[saved] theta_net_hsv.pt")
