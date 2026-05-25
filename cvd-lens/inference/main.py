from pathlib import Path
import io
import os
import tempfile
import uuid

import cv2
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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
    arr = arr.transpose(2, 0, 1)
    cvd_val = CVD_VALUES.get(cvd_type, 0.5)
    cvd_ch = np.full((1, 256, 256), cvd_val, dtype=np.float32)
    inp = np.concatenate([arr, cvd_ch], axis=0)[np.newaxis]

    output = session.run(["output"], {"input": inp})[0]
    output = np.clip(output[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(output).save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


import shutil
_FFMPEG = (
    shutil.which("ffmpeg")
    or r"C:\Users\SCH\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
)


def _correct_frame(frame_bgr: np.ndarray, cvd_val: float) -> np.ndarray:
    """BGR frame (any size) → corrected BGR frame at same size."""
    orig_h, orig_w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_256 = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)

    arr    = rgb_256.astype(np.float32) / 255.0
    arr    = arr.transpose(2, 0, 1)
    cvd_ch = np.full((1, 256, 256), cvd_val, dtype=np.float32)
    inp    = np.concatenate([arr, cvd_ch], axis=0)[np.newaxis]

    out = session.run(["output"], {"input": inp})[0]
    out = np.clip(out[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # scale back to original resolution
    if (orig_w, orig_h) != (256, 256):
        out_bgr = cv2.resize(out_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return out_bgr


@app.post("/infer/video")
async def infer_video(video: UploadFile = File(...), cvd_type: str = Form(...)):
    """
    Upload video → correct each frame → return H.264 MP4 at original resolution.
    Uses ffmpeg pipe: OpenCV reads frames, corrects via ONNX, writes raw BGR to
    ffmpeg stdin → ffmpeg encodes H.264 directly (browser-compatible).
    """
    import subprocess

    cvd_val = CVD_VALUES.get(cvd_type, 0.5)
    suffix  = Path(video.filename or "input.mp4").suffix or ".mp4"
    tmp_in  = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_in.close(); tmp_out.close()

    try:
        with open(tmp_in.name, "wb") as f:
            f.write(await video.read())

        cap = cv2.VideoCapture(tmp_in.name)
        if not cap.isOpened():
            raise ValueError("영상을 열 수 없습니다.")

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # libx264/yuv420p requires even dimensions
        enc_w = w + (w % 2)
        enc_h = h + (h % 2)

        # pipe raw BGR frames directly into ffmpeg → H.264 MP4
        proc = subprocess.Popen(
            [_FFMPEG, "-y",
             "-f", "rawvideo", "-vcodec", "rawvideo",
             "-s", f"{enc_w}x{enc_h}", "-pix_fmt", "bgr24", "-r", str(fps),
             "-i", "pipe:0",
             "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             "-movflags", "+faststart",
             tmp_out.name],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            corrected = _correct_frame(frame, cvd_val)
            # pad to even dimensions if needed
            if enc_w != w or enc_h != h:
                corrected = cv2.copyMakeBorder(corrected, 0, enc_h - h, 0, enc_w - w, cv2.BORDER_REPLICATE)
            proc.stdin.write(corrected.tobytes())
            frame_idx += 1

        cap.release()
        proc.stdin.close()
        retcode = proc.wait()
        ffmpeg_stderr = proc.stderr.read().decode("utf-8", errors="replace")
        if retcode != 0:
            raise RuntimeError(f"ffmpeg failed (code {retcode}):\n{ffmpeg_stderr[-2000:]}")

        def stream_file():
            with open(tmp_out.name, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk
            for p in [tmp_in.name, tmp_out.name]:
                try: os.unlink(p)
                except OSError: pass

        filename = f"cvdlens_{cvd_type}_{uuid.uuid4().hex[:8]}.mp4"
        return StreamingResponse(
            stream_file(),
            media_type="video/mp4",
            headers={"Content-Disposition": f'attachment; filename="{filename}"',
                     "X-Frame-Count": str(frame_idx)},
        )

    except Exception as e:
        for p in [tmp_in.name, tmp_out.name]:
            try: os.unlink(p)
            except OSError: pass
        return {"error": str(e)}
