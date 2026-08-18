"""
Local mirror of the deployed inference pipeline (cvd-lens/inference/main.py) with
LOCAL ONNX (no web round-trip) and a guided-filter on/off toggle, returning
intermediates for diagnosis. Imports the SAME canonical guided.py the server uses.

The letterbox / delta-composite steps are copied verbatim from main.py so the
before-guided path is byte-for-byte the current production behaviour.

Not a FastAPI app — a helper for the diagnostic/eval scripts (diag_crr,
diag_dilution, reeval_guided).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as rt

_REPO = Path(__file__).resolve().parent.parent
_INFER_DIR = _REPO / "cvd-lens" / "inference"
sys.path.insert(0, str(_INFER_DIR))
from guided import guided_filter          # canonical, shared with the server
import config as _server_config           # defaults for radius divisor / eps

MODEL_DIR = _INFER_DIR / "model"
MAX_SIDE = 2048
_PROVIDERS = ["CPUExecutionProvider"]
SESSIONS = {
    t: rt.InferenceSession(str(MODEL_DIR / f"cvdlens_{t}.onnx"), providers=_PROVIDERS)
    for t in ("p", "d", "t")
}

# Post-processing defaults (match the server config unless overridden per call).
DEFAULT_RADIUS_DIVISOR = _server_config.GUIDED_RADIUS_DIVISOR
DEFAULT_EPS = _server_config.GUIDED_EPS
DEFAULT_MAX_SIDE = _server_config.GUIDED_MAX_SIDE


def cap_long_side(img_f32, max_side=MAX_SIDE):
    h, w = img_f32.shape[:2]
    m = max(h, w)
    if m > max_side:
        s = max_side / m
        img_f32 = cv2.resize(img_f32, (round(w * s), round(h * s)),
                             interpolation=cv2.INTER_AREA)
    return img_f32


def _run_float(rgb256, cvd_type, severity):
    chw = rgb256.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    sev = np.array([[severity]], dtype=np.float32)
    out = SESSIONS[cvd_type].run(["out_srgb"], {"srgb": chw, "severity": sev})[0]
    return out[0].transpose(1, 2, 0).astype(np.float32)


def _letterbox(img_f32, size=256):
    h, w = img_f32.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img_f32, (nw, nh), interpolation=interp)
    left, top = (size - nw) // 2, (size - nh) // 2
    right, bottom = size - nw - left, size - nh - top
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return canvas, (left, top, left + nw, top + nh)


def correct(img_f32, cvd_type, severity, *, use_guided,
            radius_divisor=None, eps=None):
    """Native-resolution corrected image + intermediates.

    Returns dict with:
        corrected      (H,W,3) final (clipped) output
        delta_pre      (H,W,3) upsampled delta BEFORE guided filter
        delta_post     (H,W,3) delta actually added (== delta_pre if not guided)
        delta256       (256,256,3) raw model delta (out - letterbox)
        box            (x0,y0,x1,y1) content box in 256-space
        radius, eps    the guided-filter params used (radius None if off)
    """
    h, w = img_f32.shape[:2]
    lb, (x0, y0, x1, y1) = _letterbox(img_f32, 256)
    out = _run_float(lb, cvd_type, severity)
    delta256 = out - lb
    delta_pre = cv2.resize(delta256[y0:y1, x0:x1], (w, h),
                           interpolation=cv2.INTER_LINEAR)

    radius = None
    if use_guided:
        rdiv = DEFAULT_RADIUS_DIVISOR if radius_divisor is None else radius_divisor
        e = DEFAULT_EPS if eps is None else eps
        radius = max(1, max(h, w) // rdiv)
        delta_post = guided_filter(img_f32, delta_pre, radius, e, max_side=DEFAULT_MAX_SIDE)
    else:
        e = None
        delta_post = delta_pre

    corrected = np.clip(img_f32 + delta_post, 0.0, 1.0)
    return dict(corrected=corrected, delta_pre=delta_pre, delta_post=delta_post,
                delta256=delta256, box=(x0, y0, x1, y1), radius=radius, eps=e)


def load_rgb(path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return cap_long_side(img)
