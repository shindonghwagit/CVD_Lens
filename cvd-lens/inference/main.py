from pathlib import Path
import io
import os
import shutil
import tempfile
import uuid

import cv2
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

import config
from guided import guided_filter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Phase 1 model_best (step 9000): one self-contained graph per CVD type. ──
# Inputs: srgb (1,3,256,256) float32, severity (1,1) float32.  Output: out_srgb.
# Replaces the pre-pivot 4-channel cvdlens_fp32.onnx.
MODEL_DIR = Path(__file__).parent / "model"
_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
SESSIONS = {
    t: rt.InferenceSession(str(MODEL_DIR / f"cvdlens_{t}.onnx"), providers=_PROVIDERS)
    for t in ("p", "d", "t")
}
DEFAULT_TYPE = "d"

# Longest edge of the returned image. The delta-composite path below returns at
# native resolution, so this caps response size for very large uploads (4K etc.).
MAX_SIDE = 2048


def _run_float(rgb256: np.ndarray, cvd_type: str, severity: float) -> np.ndarray:
    """rgb256: (256,256,3) float32 in [0,1] → raw model output (256,256,3) float32.

    No clipping — the caller needs the raw correction so `out - in` recovers the
    true delta before compositing.
    """
    sess = SESSIONS.get(cvd_type, SESSIONS[DEFAULT_TYPE])
    chw = rgb256.transpose(2, 0, 1)[np.newaxis].astype(np.float32)   # (1,3,256,256)
    sev = np.array([[severity]], dtype=np.float32)                   # (1,1)
    out = sess.run(["out_srgb"], {"srgb": chw, "severity": sev})[0]
    return out[0].transpose(1, 2, 0).astype(np.float32)


def _letterbox(img_f32: np.ndarray, size: int = 256):
    """Aspect-preserving resize into size×size with edge-replicate padding.

    Returns (canvas float32 [0,1], (x0,y0,x1,y1) content-box in canvas coords).
    Replaces the old center-crop so no field of view is thrown away, and the
    padding is a benign replicate of the border (its delta is discarded anyway).
    """
    h, w = img_f32.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img_f32, (nw, nh), interpolation=interp)
    left, top = (size - nw) // 2, (size - nh) // 2
    right, bottom = size - nw - left, size - nh - top
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return canvas, (left, top, left + nw, top + nh)


def _correct_image(img_f32: np.ndarray, cvd_type: str, severity: float) -> np.ndarray:
    """img_f32: (H,W,3) float32 [0,1] at native resolution → corrected, same shape.

    Bilateral-grid delta-composite: infer the color correction at 256, take the
    delta (out − in) over the content box only, bilinear-upsample it to native
    resolution, and add it back to the *original* pixels. The correction is a
    smooth low-frequency color shift, so upsampling it loses nothing while the
    original's high-frequency detail (text edges etc.) is preserved.
    """
    h, w = img_f32.shape[:2]
    lb, (x0, y0, x1, y1) = _letterbox(img_f32, 256)
    out = _run_float(lb, cvd_type, severity)
    delta = (out - lb)[y0:y1, x0:x1]                      # content-box delta only
    delta_full = cv2.resize(delta, (w, h), interpolation=cv2.INTER_LINEAR)

    # Guided-filter post-processing: snap the delta to the original's edges
    # (region boundaries sharpen) and flatten it inside uniform-guide regions
    # (removes the low-frequency delta gradient), using the original as guide.
    # Model is untouched — this operates on the composited delta only.
    if config.GUIDED_FILTER_ENABLED:
        radius = max(1, max(h, w) // config.GUIDED_RADIUS_DIVISOR)
        delta_full = guided_filter(img_f32, delta_full, radius, config.GUIDED_EPS,
                                   max_side=config.GUIDED_MAX_SIDE)

    return np.clip(img_f32 + delta_full, 0.0, 1.0)


def _cap_long_side(img_f32: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img_f32.shape[:2]
    m = max(h, w)
    if m > max_side:
        s = max_side / m
        img_f32 = cv2.resize(img_f32, (round(w * s), round(h * s)),
                             interpolation=cv2.INTER_AREA)
    return img_f32


def _to_u8(img_f32: np.ndarray) -> np.ndarray:
    """Single float→uint8 hop, with rounding (avoids truncation banding)."""
    return np.clip(img_f32 * 255.0 + 0.5, 0, 255).astype(np.uint8)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    cvd_type: str = Form(...),
    severity: float = Form(1.0),   # optional; older frontend omits → 1.0
):
    data = await image.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    arr = _cap_long_side(np.asarray(img, dtype=np.float32) / 255.0)
    out = _to_u8(_correct_image(arr, cvd_type, severity))

    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="JPEG", quality=config.RESPONSE_JPEG_QUALITY)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


_FFMPEG = (
    shutil.which("ffmpeg")
    or r"C:\Users\SCH\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
)


def _correct_frame(frame_bgr: np.ndarray, cvd_type: str, severity: float = 1.0) -> np.ndarray:
    """BGR frame (any size) → corrected BGR frame at the same size.

    Same delta-composite path as /infer. No long-side cap here: the ffmpeg
    encoder is sized to the capture resolution, and the composite preserves it.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = _correct_image(rgb, cvd_type, severity)
    return cv2.cvtColor(_to_u8(out), cv2.COLOR_RGB2BGR)


@app.post("/infer/video")
async def infer_video(
    video: UploadFile = File(...),
    cvd_type: str = Form(...),
    severity: float = Form(1.0),
):
    import subprocess
    import threading

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
        enc_w = w + (w % 2)
        enc_h = h + (h % 2)

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

        # Drain stderr on a thread. ffmpeg's stderr pipe is only ~64 KB; if we
        # let it fill (reading it only after wait()), ffmpeg blocks on it, stops
        # consuming stdin, and stdin.write() deadlocks. This is why a straight
        # read-after-wait hangs on any real-length video.
        err_chunks: list[bytes] = []
        err_thread = threading.Thread(target=lambda: err_chunks.append(proc.stderr.read()), daemon=True)
        err_thread.start()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            corrected = _correct_frame(frame, cvd_type, severity)
            if enc_w != w or enc_h != h:
                corrected = cv2.copyMakeBorder(corrected, 0, enc_h - h, 0, enc_w - w, cv2.BORDER_REPLICATE)
            proc.stdin.write(corrected.tobytes())
            frame_idx += 1

        cap.release()
        proc.stdin.close()
        retcode = proc.wait()
        err_thread.join(timeout=5)
        ffmpeg_stderr = b"".join(err_chunks).decode("utf-8", errors="replace")
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
