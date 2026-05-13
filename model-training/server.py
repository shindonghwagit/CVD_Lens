"""
CVDLens 추론 서버 (FastAPI + PyTorch)

사용법:
    py server.py --ckpt C:/Users/SCH/Downloads/best-epoch=026-val_loss=0.0265.ckpt
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import CVDLitModule

CVD_VALUES = {"p": 0.0, "d": 0.5, "t": 1.0}

app = FastAPI(title="CVDLens API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 전역 모델 (서버 시작 시 1회 로드) ──────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
net: torch.nn.Module | None = None


def load_model(ckpt_path: str):
    global net
    print(f"모델 로드: {ckpt_path}")
    print(f"디바이스: {device}")
    module = CVDLitModule.load_from_checkpoint(ckpt_path, map_location=device)
    module.eval()
    net = module.model
    print("모델 로드 완료")


# ── 엔드포인트 ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": device, "model_loaded": net is not None}


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    cvd_type: str = Form(...),
):
    assert net is not None, "모델 미로드"
    assert cvd_type in CVD_VALUES, f"잘못된 cvd_type: {cvd_type}"

    # ── 이미지 로드 ────────────────────────────────────────
    img_bytes = await image.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_array = np.array(pil)  # (H, W, 3) uint8
    h, w = img_array.shape[:2]

    # ── 전처리 → (1, 4, H, W) ─────────────────────────────
    img_t = torch.from_numpy(img_array).float().permute(2, 0, 1) / 255.0  # (3, H, W)
    cvd_ch = torch.full((1, h, w), CVD_VALUES[cvd_type])                   # (1, H, W)
    inp = torch.cat([img_t, cvd_ch], dim=0).unsqueeze(0).to(device)        # (1, 4, H, W)

    # ── 추론 ──────────────────────────────────────────────
    with torch.no_grad():
        out = net(inp).squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

    # ── 후처리 → JPEG 응답 ─────────────────────────────────
    out_img = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out_img).save(buf, format="JPEG", quality=85)
    return Response(buf.getvalue(), media_type="image/jpeg")


# ── 진입점 ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="체크포인트 경로")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.ckpt)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
