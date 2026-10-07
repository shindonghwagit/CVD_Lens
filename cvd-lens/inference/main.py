from pathlib import Path
import io
import logging
import os
import shutil
import tempfile
import uuid

import cv2
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image

import config
from guided import guided_filter

app = FastAPI()
logger = logging.getLogger("cvdlens.inference")

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CVDLENS_ALLOWED_ORIGINS",
        "http://localhost:3000,https://cvd-lens.vercel.app",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Phase 1 model_best (step 9000): P/D self-contained graphs. ──
# Inputs: srgb (1,3,256,256) float32, severity (1,1) float32.  Output: out_srgb.
# Replaces the pre-pivot 4-channel cvdlens_fp32.onnx.
MODEL_DIR = Path(__file__).parent / "model"
_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
SESSIONS = {
    t: rt.InferenceSession(str(MODEL_DIR / f"cvdlens_{t}.onnx"), providers=_PROVIDERS)
    for t in ("p", "d")
}

# Longest edge of the returned image. The delta-composite path below returns at
# native resolution, so this caps response size for very large uploads (4K etc.).
MAX_SIDE = 2048
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_VIDEO_BYTES = 250 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
MAX_VIDEO_PIXELS = 1920 * 1080
MAX_VIDEO_SECONDS = 5 * 60
VALID_CVD_TYPES = {"p", "d", "t"}


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="업로드 파일이 너무 큽니다.")
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_cvd_type(cvd_type: str) -> None:
    if cvd_type not in VALID_CVD_TYPES:
        raise HTTPException(status_code=422, detail="지원하지 않는 색각 유형입니다.")


def _run_float(rgb256: np.ndarray, cvd_type: str, severity: float) -> np.ndarray:
    """rgb256: (256,256,3) float32 in [0,1] → raw model output (256,256,3) float32.

    No clipping — the caller needs the raw correction so `out - in` recovers the
    true delta before compositing.
    """
    sess = SESSIONS[cvd_type]
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


def _band(x: np.ndarray, lo: float, hi: float, w: float = 12.0) -> np.ndarray:
    """Soft in-range indicator on [lo,hi] with `w`-wide ramps at both ends."""
    return np.clip((x - lo) / w, 0.0, 1.0) * np.clip((hi - x) / w, 0.0, 1.0)


# Tritan hue-rotation base angle at severity 1.0 (OpenCV H units, 0..179 == 0..360°).
_TRITAN_BASE_DEG = 30.0


def _tritan_hue_shift_fixed_legacy(img_f32: np.ndarray, severity: float) -> np.ndarray:
    """Analytic, saturation-preserving hue rotation for tritan (blue↔yellow axis).

    Rotates blue (H~90–135) toward violet and yellow (H~18–40) toward yellow-green,
    keeping S and V fixed — so blue/yellow move off the tritan confusion axis and
    become distinguishable WITHOUT the desaturation ("물빠짐") the learned model
    produced by adding the opponent channel. Red/green/gray are outside both hue
    bands (and below the saturation floor for gray) → untouched (selectivity).
    Validated on the tritan_blue test-set (cvdlens_v2/tritan_hue_method.py):
    coverage/CRR/selectivity pass, |Δsat|≲0.03. Severity scales the angle.
    """
    deg = _TRITAN_BASE_DEG * float(np.clip(severity, 0.0, 1.0))
    hsv = cv2.cvtColor(_to_u8(img_f32), cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat_g = np.clip((S - 18.0) / 50.0, 0.0, 1.0)              # exclude near-gray
    g_blue = sat_g * _band(H, 90.0, 135.0)                    # blue
    g_yellow = sat_g * _band(H, 18.0, 40.0)                   # yellow
    hsv[..., 0] = (H + (g_blue + g_yellow) * deg) % 180.0     # S,V untouched
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0


_TRITAN_ANGLES = (-36.0, -30.0, -24.0, -18.0, -12.0, 0.0,
                  12.0, 18.0, 24.0, 30.0, 36.0)
_TRITAN_RGB2LMS = np.array([
    [17.8824, 43.5161, 4.11935], [3.45565, 27.1554, 3.86714],
    [.02996, .18431, 1.46720],
], np.float32)
_TRITAN_LMS2RGB = np.linalg.inv(_TRITAN_RGB2LMS).astype(np.float32)
_TRITAN_SIM = _TRITAN_LMS2RGB @ np.array(
    [[1, 0, 0], [0, 1, 0], [-.395913, .801109, 0]], np.float32
) @ _TRITAN_RGB2LMS
_TRITAN_MACHADO = np.array([
    [1.255528, -.076749, -.178779],
    [-.078411, .930809, .147602],
    [.004733, .691367, .303900],
], np.float32)
_RGB2XYZ = np.array([
    [.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750],
    [.0193339, .1191920, .9503041],
], np.float32)
_D65 = np.array([.95047, 1.0, 1.08883], np.float32)


def _tritan_apply_angle(img_f32: np.ndarray, angle: float) -> np.ndarray:
    """Saturation/value-preserving gated hue shift; zero is exact identity."""
    if angle == 0.0:
        return img_f32.copy()
    hsv = cv2.cvtColor(_to_u8(img_f32), cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, saturation = hsv[..., 0], hsv[..., 1]
    sat_gate = np.clip((saturation - 18.0) / 50.0, 0.0, 1.0)
    gate = sat_gate * (_band(hue, 90.0, 135.0) + _band(hue, 18.0, 40.0))
    hsv[..., 0] = (hue + gate * angle) % 180.0
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0


def _tritan_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x > .04045, ((np.maximum(x, 0) + .055) / 1.055) ** 2.4,
                    x / 12.92).astype(np.float32)


def _tritan_lab(x_linear: np.ndarray) -> np.ndarray:
    xyz = np.einsum("ij,hwj->hwi", _RGB2XYZ, x_linear) / _D65
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(np.maximum(xyz, 0)),
                 xyz / (3 * d * d) + 4.0 / 29.0)
    return np.stack((116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), axis=-1)


def _tritan_simulate(x_linear: np.ndarray, matrix: np.ndarray = _TRITAN_SIM) -> np.ndarray:
    return np.clip(np.einsum("ij,hwj->hwi", matrix, x_linear), 0, 1)


def _tritan_probe_score(original: np.ndarray, candidate: np.ndarray,
                        matrix: np.ndarray = _TRITAN_SIM) -> float:
    orig_linear = _tritan_linear(original)
    sim_orig = _tritan_simulate(orig_linear, matrix)
    sim_out = _tritan_simulate(_tritan_linear(candidate), matrix)
    delta_e = np.linalg.norm(_tritan_lab(orig_linear) - _tritan_lab(sim_orig), axis=2)
    weight = np.clip((delta_e - 12.0) / 18.0, 0, 1).astype(np.float32)
    weight = cv2.GaussianBlur(weight, (11, 11), 3, borderType=cv2.BORDER_REFLECT)
    if float(weight.mean()) < 1e-6:
        return 1.0
    before = cv2.GaussianBlur(sim_orig, (5, 5), 1, borderType=cv2.BORDER_REFLECT)
    after = cv2.GaussianBlur(sim_out, (5, 5), 1, borderType=cv2.BORDER_REFLECT)

    def gradient(x: np.ndarray) -> np.ndarray:
        dx = np.pad(np.abs(x[:, 1:] - x[:, :-1]).mean(2), ((0, 0), (0, 1)))
        dy = np.pad(np.abs(x[1:] - x[:-1]).mean(2), ((0, 1), (0, 0)))
        return dx + dy

    denominator = float((weight * gradient(before)).sum() / (weight.sum() + 1e-8))
    numerator = float((weight * gradient(after)).sum() / (weight.sum() + 1e-8))
    return numerator / (denominator + 1e-8)


def _tritan_hue_shift(img_f32: np.ndarray, severity: float) -> np.ndarray:
    """Select a per-image hue angle that improves Brettel-view contrast."""
    h, w = img_f32.shape[:2]
    scale = min(1.0, 512.0 / max(h, w))
    probe = (cv2.resize(img_f32, (round(w * scale), round(h * scale)),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else img_f32)
    strength = float(np.clip(severity, 0.0, 1.0))
    scored = []
    for base_angle in _TRITAN_ANGLES:
        angle = base_angle * strength
        candidate = _tritan_apply_angle(probe, angle)
        brettel = _tritan_probe_score(probe, candidate, _TRITAN_SIM)
        machado = _tritan_probe_score(probe, candidate, _TRITAN_MACHADO)
        scored.append((min(brettel, machado), angle))
    best_score, best_angle = max(scored, key=lambda item: item[0])
    if best_score < 1.002:
        best_angle = 0.0
    return _tritan_apply_angle(img_f32, best_angle)


def _correct_image(img_f32: np.ndarray, cvd_type: str, severity: float) -> np.ndarray:
    """img_f32: (H,W,3) float32 [0,1] at native resolution → corrected, same shape.

    Protan/deutan: bilateral-grid delta-composite — infer the color correction at
    256, take the delta (out − in) over the content box, bilinear-upsample and add
    it back to the *original* pixels (low-frequency shift, detail preserved).

    Tritan: dedicated saturation-preserving hue rotation (see _tritan_hue_shift),
    computed at native resolution. Both paths share the guided-filter smoothing +
    composite below.
    """
    h, w = img_f32.shape[:2]
    if cvd_type == "t":
        delta_full = _tritan_hue_shift(img_f32, severity) - img_f32
    else:
        lb, (x0, y0, x1, y1) = _letterbox(img_f32, 256)
        # P/D checkpoints were trained only at severity=1.0. Keep the network
        # on-distribution and treat severity as an output-delta strength.
        out = _run_float(lb, cvd_type, 1.0)
        delta = ((out - lb) * severity)[y0:y1, x0:x1]     # content-box delta only
        delta_full = cv2.resize(delta, (w, h), interpolation=cv2.INTER_LINEAR)

    # Guided-filter post-processing: snap the delta to the original's edges
    # (region boundaries sharpen) and flatten it inside uniform-guide regions
    # (removes the low-frequency delta gradient), using the original as guide.
    # Model is untouched — this operates on the composited delta only.
    if config.GUIDED_FILTER_ENABLED and cvd_type != "t":
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
):
    _validate_cvd_type(cvd_type)
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="이미지 파일만 업로드할 수 있습니다.")
    data = await _read_limited(image, MAX_IMAGE_BYTES)
    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="이미지 해상도가 너무 큽니다.")
            img = opened.convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.") from exc

    arr = _cap_long_side(np.asarray(img, dtype=np.float32) / 255.0)
    out = _to_u8(_correct_image(arr, cvd_type, 1.0))

    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="JPEG", quality=config.RESPONSE_JPEG_QUALITY)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


# ffmpeg는 PATH에서만 찾는다. 컨테이너엔 Dockerfile이 설치. 없으면 /infer/video가 500 반환.
# (예전엔 로컬 Windows 경로로 폴백해 배포 서버에서 FileNotFoundError → 검은 영상으로 실패했음.)
_FFMPEG = shutil.which("ffmpeg")


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
):
    import subprocess
    import threading

    _validate_cvd_type(cvd_type)
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="영상 파일만 업로드할 수 있습니다.")
    if _FFMPEG is None:
        return JSONResponse(status_code=500,
                            content={"error": "서버에 ffmpeg가 설치되어 있지 않아 영상 보정을 처리할 수 없습니다."})

    suffix  = Path(video.filename or "input.mp4").suffix or ".mp4"
    tmp_in  = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_in.close(); tmp_out.close()

    try:
        with open(tmp_in.name, "wb") as f:
            f.write(await _read_limited(video, MAX_VIDEO_BYTES))

        cap = cv2.VideoCapture(tmp_in.name)
        if not cap.isOpened():
            raise ValueError("영상을 열 수 없습니다.")

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if w <= 0 or h <= 0 or w * h > MAX_VIDEO_PIXELS:
            cap.release()
            raise HTTPException(status_code=413, detail="영상 해상도는 최대 1920×1080입니다.")
        if frame_count > 0 and frame_count / fps > MAX_VIDEO_SECONDS:
            cap.release()
            raise HTTPException(status_code=413, detail="영상 길이는 최대 5분입니다.")
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
            corrected = _correct_frame(frame, cvd_type, 1.0)
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

    except HTTPException:
        for p in [tmp_in.name, tmp_out.name]:
            try: os.unlink(p)
            except OSError: pass
        raise
    except Exception:
        logger.exception("Video inference failed")
        for p in [tmp_in.name, tmp_out.name]:
            try: os.unlink(p)
            except OSError: pass
        return JSONResponse(status_code=500, content={"error": "영상 보정 처리에 실패했습니다."})
