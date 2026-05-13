from pathlib import Path
import io

import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "model" / "cvdlens_fp32.onnx"
session = rt.InferenceSession(
    str(MODEL_PATH),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

CVD_VALUES = {"p": 0.0, "d": 0.5, "t": 1.0}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/infer")
async def infer(image: UploadFile = File(...), cvd_type: str = Form(...)):
    data = await image.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    img = img.resize((256, 256), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # (3, 256, 256)

    cvd_val = CVD_VALUES.get(cvd_type, 0.5)
    cvd_ch = np.full((1, 256, 256), cvd_val, dtype=np.float32)
    inp = np.concatenate([arr, cvd_ch], axis=0)[np.newaxis]  # (1, 4, 256, 256)

    output = session.run(["output"], {"input": inp})[0]  # (1, 3, 256, 256)
    output = np.clip(output[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(output).save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")
