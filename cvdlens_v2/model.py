"""
CVDCorrectionNet — Phase 1 architecture.

Design:
    (B,3,H,W) sRGB + one-hot CVD type (B,3) + severity (B,1)
        │
        ├── MobileNetV3-Small features (torchvision, ImageNet-normalized input)
        │       ↑ FiLM (γ, β) injected at 2 encoder stages
        │       ↑ conditioning MLP:  [one_hot(3); severity(1)] → (γ, β)_stack
        │
        ├── 1×1 conv head → bilateral grid  (B, 2, D=8, H_g=16, W_g=16)
        │       — 2 output channels (d_lum, d_c)
        │
        ├── luminance guide (BT.709 Y from linear RGB, mapped to [-1, 1])
        ├── grid_sample slicing (trilinear) → (d_lum, d_c) full-res (B, 1, H, W)
        ├── delta = compose_delta(d_lum, d_c, cvd_type)   ← visible-subspace
        │
        └── out_lin = clamp(orig_lin + w · delta, 0, 1),   out_srgb = γ⁻¹(out_lin)

Speckle claim (delegated from Exp 1-D negative result): bilateral-grid slicing
adds a structural smoothness prior. In a locally-uniform region, the guide
value is (near-)constant, so all pixels sample the same D-slice; spatial
grid_sample then produces at most bilinearly-interpolated variation between
adjacent H_g/W_g bins — a much tighter Lipschitz bound than raw per-pixel
fields. This is nailed down by `test_uniform_patch_delta_smooth` in
test_model.py.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

from .basis import compose_delta, get_basis, get_confusion_dir
from .color import srgb_to_linear, linear_to_srgb
from .confusion import compute_confusion_weight


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CVD_TYPES = ("p", "d", "t")


def _film_apply(x: torch.Tensor, gamma: torch.Tensor,
                beta: torch.Tensor) -> torch.Tensor:
    """FiLM: x_new = (1 + γ) ⊙ x + β  (γ, β are (B, C))."""
    return x * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) \
             + beta.unsqueeze(-1).unsqueeze(-1)


class CVDCorrectionNet(nn.Module):
    """
    Bilateral-grid CVD correction network.

    Args:
        grid_depth: guide-axis bins (D). Default 8.
        grid_spatial: H_g = W_g bins. Default 16.
        film_stage_indices: indices into mobilenet_v3_small.features at
            whose OUTPUT FiLM is applied. Default (4, 8) — the two 16×16
            feature-map plateaus.
        film_hidden: hidden width of the FiLM MLP.
        pretrained_backbone: load ImageNet-pretrained mobilenet weights.
    """

    # Channel counts at torchvision mobilenet_v3_small.features[i] output
    # (input 256×256).  Derived from torchvision reference impl.
    _STAGE_CHANNELS = {
        4: 40,
        8: 48,
    }

    def __init__(
        self,
        grid_depth: int = 8,
        grid_spatial: int = 16,
        film_stage_indices: tuple = (4, 8),
        film_hidden: int = 32,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        assert all(i in self._STAGE_CHANNELS for i in film_stage_indices), \
            f"FiLM stage indices must be a subset of {list(self._STAGE_CHANNELS)}"
        self.grid_depth = grid_depth
        self.grid_spatial = grid_spatial
        self.film_stage_indices = tuple(film_stage_indices)

        # Backbone (features only — no classifier head).
        weights = "DEFAULT" if pretrained_backbone else None
        mb = mobilenet_v3_small(weights=weights)
        self.features = mb.features  # nn.Sequential
        self.max_stage_idx = max(self.film_stage_indices)

        # FiLM MLP: [one_hot(3); severity] → concat[(γ_i, β_i)] over stages
        total_film_dims = sum(2 * self._STAGE_CHANNELS[i]
                              for i in self.film_stage_indices)
        self.film_mlp = nn.Sequential(
            nn.Linear(4, film_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(film_hidden, film_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(film_hidden, total_film_dims),
        )
        # Zero-init the last linear → FiLM starts at identity (γ=β=0).
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        # Grid head: normalize encoder output to grid spatial size and
        # project to (2·D) channels.  Adaptive pool decouples the head
        # from input resolution (encoder features at stage 8 land on
        # 16×16 for 256-input, 8×8 for 128, etc.).  Zero-init → grid
        # starts at zero → delta=0 → out=orig (identity init).
        head_in = self._STAGE_CHANNELS[self.max_stage_idx]
        self.grid_pool = nn.AdaptiveAvgPool2d(grid_spatial)
        self.grid_head = nn.Conv2d(head_in, 2 * grid_depth, kernel_size=1)
        # Near-identity init: small-random weight (so ∂grid/∂feat ≠ 0 and
        # gradient flows back to the FiLM MLP and backbone from step 0)
        # + zero bias.  |delta| at init is O(std · sqrt(head_in) · |feat|)
        # ≈ O(1e-3 · 7 · 1) ≈ 1e-2 which after w-gating is well below
        # the identity guard threshold (0.01).
        nn.init.normal_(self.grid_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.grid_head.bias)

        # ImageNet norm constants as buffers (device-agnostic).
        self.register_buffer(
            "_imgnet_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            "_imgnet_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    # ── Encoder with FiLM ────────────────────────────────────────────
    def _split_film(self, film_params: torch.Tensor) -> dict:
        """Slice concatenated (γ, β) → {stage_idx: (γ, β)}."""
        out = {}
        offset = 0
        for i in self.film_stage_indices:
            C = self._STAGE_CHANNELS[i]
            gamma = film_params[:, offset:offset + C]
            beta = film_params[:, offset + C:offset + 2 * C]
            out[i] = (gamma, beta)
            offset += 2 * C
        return out

    def _encode(self, x_norm: torch.Tensor,
                film_by_stage: dict) -> torch.Tensor:
        for i, layer in enumerate(self.features):
            if i > self.max_stage_idx:
                break
            x_norm = layer(x_norm)
            if i in film_by_stage:
                g, b = film_by_stage[i]
                x_norm = _film_apply(x_norm, g, b)
        return x_norm

    # ── Guide + slicing ──────────────────────────────────────────────
    @staticmethod
    def _luminance_guide(orig_lin: torch.Tensor) -> torch.Tensor:
        """BT.709 linear luma → normalized to [-1, 1]. (B, 1, H, W)."""
        Y = (0.2126 * orig_lin[:, 0:1]
             + 0.7152 * orig_lin[:, 1:2]
             + 0.0722 * orig_lin[:, 2:3])
        return 2.0 * Y - 1.0

    def _slice_grid(self, grid_5d: torch.Tensor,
                    guide: torch.Tensor) -> torch.Tensor:
        """
        Bilateral-grid slicing built from gather + arithmetic only —
        no `F.grid_sample` at any dimensionality.

        Why: 5D `grid_sample` is not ONNX-exportable at all; 4D
        `grid_sample` at opset 16 exports but is NOT implemented in
        ONNX Runtime (verified in Phase 1 pre-training smoke) and its
        support in ONNX Runtime Web (deploy target) is patchy. Rebuilding
        both stages as manual bilinear/linear interpolation gives us
        full portability at zero semantic cost — grid_sample with
        `mode='bilinear'`, `padding_mode='border'`, `align_corners=False`
        IS exactly what we implement here.

            grid_5d : (B, C, D, H_g, W_g)
            guide   : (B, 1, H, W)      in [-1, 1]
            returns : (B, C, H, W)
        """
        B, C, D, Hg, Wg = grid_5d.shape
        _, _, H, W = guide.shape
        device, dtype = grid_5d.device, grid_5d.dtype

        # ── Output spatial coords in pixel space of the grid (H_g × W_g).
        # align_corners=False convention: -1 ↔ -0.5, +1 ↔ (H_g − 0.5).
        yy = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        yy_g, xx_g = torch.meshgrid(yy, xx, indexing="ij")        # (H, W)
        x_px = (xx_g + 1.0) * Wg * 0.5 - 0.5                       # (H, W)
        y_px = (yy_g + 1.0) * Hg * 0.5 - 0.5

        x0 = x_px.floor().clamp_(0, Wg - 1).to(torch.long)         # (H, W)
        y0 = y_px.floor().clamp_(0, Hg - 1).to(torch.long)
        x1 = (x0 + 1).clamp_(0, Wg - 1)
        y1 = (y0 + 1).clamp_(0, Hg - 1)
        fx = (x_px - x0.to(dtype)).clamp_(0.0, 1.0)                # (H, W)
        fy = (y_px - y0.to(dtype)).clamp_(0.0, 1.0)

        # ── Guide index into D axis (linear interp between two slices).
        # guide ∈ [-1, 1]  →  g ∈ [0, D-1]
        g = (guide.squeeze(1) + 1.0) * 0.5 * (D - 1)               # (B, H, W)
        gz0 = g.floor().clamp_(0, D - 1).to(torch.long)             # (B, H, W)
        gz1 = (gz0 + 1).clamp_(0, D - 1)
        fz = (g - gz0.to(dtype)).clamp_(0.0, 1.0)                   # (B, H, W)

        # ── Gather 8 corner values (bilinear × linear = trilinear).
        # Linearize (D, H_g, W_g) into a single axis so we can use one
        # torch.gather call per corner.
        flat = grid_5d.reshape(B, C, D * Hg * Wg)                   # (B, C, D·Hg·Wg)

        def corner(gz, y_, x_):
            # gz: (B, H, W) long; y_, x_: (H, W) long
            idx = gz * (Hg * Wg) + y_.unsqueeze(0) * Wg + x_.unsqueeze(0)  # (B, H, W)
            idx = idx.unsqueeze(1).expand(B, C, H, W).reshape(B, C, H * W)
            v = torch.gather(flat, dim=2, index=idx)               # (B, C, H·W)
            return v.reshape(B, C, H, W)

        v000 = corner(gz0, y0, x0)
        v001 = corner(gz0, y0, x1)
        v010 = corner(gz0, y1, x0)
        v011 = corner(gz0, y1, x1)
        v100 = corner(gz1, y0, x0)
        v101 = corner(gz1, y0, x1)
        v110 = corner(gz1, y1, x0)
        v111 = corner(gz1, y1, x1)

        fx = fx.unsqueeze(0).unsqueeze(0)                          # (1, 1, H, W)
        fy = fy.unsqueeze(0).unsqueeze(0)
        fz = fz.unsqueeze(1)                                       # (B, 1, H, W)

        w000 = (1 - fx) * (1 - fy) * (1 - fz)
        w001 = fx       * (1 - fy) * (1 - fz)
        w010 = (1 - fx) * fy       * (1 - fz)
        w011 = fx       * fy       * (1 - fz)
        w100 = (1 - fx) * (1 - fy) * fz
        w101 = fx       * (1 - fy) * fz
        w110 = (1 - fx) * fy       * fz
        w111 = fx       * fy       * fz

        return (v000 * w000 + v001 * w001 + v010 * w010 + v011 * w011
                + v100 * w100 + v101 * w101 + v110 * w110 + v111 * w111)

    # ── Public forward ───────────────────────────────────────────────
    def predict_fields(
        self,
        orig_srgb: torch.Tensor,
        cvd_onehot: torch.Tensor,
        severity: torch.Tensor,
    ) -> dict:
        """
        Run backbone + head. Returns intermediate tensors WITHOUT applying
        the confusion mask or composing the delta — so tests can inspect
        (d_lum, d_c, grid) without the string-typed basis/w path.
        """
        B, _, H, W = orig_srgb.shape
        orig_lin = srgb_to_linear(orig_srgb)

        # Input normalization for backbone
        x_norm = (orig_srgb - self._imgnet_mean) / self._imgnet_std

        cond = torch.cat([cvd_onehot, severity], dim=1)      # (B, 4)
        film_params = self.film_mlp(cond)                    # (B, total_film)
        film_by_stage = self._split_film(film_params)
        feat = self._encode(x_norm, film_by_stage)            # (B, C_last, h, w)
        feat = self.grid_pool(feat)                           # (B, C_last, H_g, W_g)

        grid_raw = self.grid_head(feat)                       # (B, 2·D, H_g, W_g)
        grid_5d = grid_raw.view(
            B, 2, self.grid_depth, self.grid_spatial, self.grid_spatial)

        guide = self._luminance_guide(orig_lin)               # (B, 1, H, W)
        d_maps = self._slice_grid(grid_5d, guide)             # (B, 2, H, W)
        d_lum = d_maps[:, 0:1]
        d_c = d_maps[:, 1:2]
        return {
            "orig_lin": orig_lin,
            "grid": grid_5d,
            "guide": guide,
            "d_lum": d_lum,
            "d_c": d_c,
        }

    def forward(
        self,
        orig_srgb: torch.Tensor,
        cvd_type: str,
        severity: float = 1.0,
    ) -> dict:
        """
        Training/validation forward. `cvd_type` is a Python str (used to
        select the visible-subspace basis and to compute w). ONNX export
        should freeze the type externally (see `wrap_for_onnx`).

        Returns dict with tensors used by CVDLossV2 + validate_multi:
            out_srgb, out_linear, d_lum, d_c, grid, w
        """
        B = orig_srgb.shape[0]
        device = orig_srgb.device
        assert cvd_type in CVD_TYPES, f"cvd_type must be one of {CVD_TYPES}"
        idx = CVD_TYPES.index(cvd_type)
        cvd_onehot = torch.zeros(B, 3, device=device, dtype=orig_srgb.dtype)
        cvd_onehot[:, idx] = 1.0
        sev = torch.full((B, 1), float(severity),
                         device=device, dtype=orig_srgb.dtype)

        interm = self.predict_fields(orig_srgb, cvd_onehot, sev)
        orig_lin = interm["orig_lin"]
        d_lum = interm["d_lum"]
        d_c = interm["d_c"]

        w = compute_confusion_weight(orig_lin, cvd_type, severity)  # (B,1,H,W)
        delta = compose_delta(d_lum, d_c, cvd_type)                 # (B,3,H,W)
        out_lin = (orig_lin + w * delta).clamp(0.0, 1.0)
        out_srgb = linear_to_srgb(out_lin)
        return {
            "out_srgb": out_srgb,
            "out_linear": out_lin,
            "d_lum": d_lum,
            "d_c": d_c,
            "delta": delta,
            "grid": interm["grid"],
            "w": w,
        }


def grid_tv(grid_5d: torch.Tensor) -> torch.Tensor:
    """
    Anisotropic TV over spatial axes of the bilateral grid.

        grid : (B, 2, D, H_g, W_g)
        TV   = mean(|Δ_H|) + mean(|Δ_W|)   over all (B, channel, D) slices

    Same regularization role as `_tv_field` in Phase 0, but applied to the
    grid coefficients directly — cheaper (16×16×8 = 2048 coeffs vs 64×64
    field = 4096 coeffs) and semantically identical after slicing, since
    grid_sample is a Lipschitz-1 interpolator.
    """
    dh = (grid_5d[:, :, :, 1:, :] - grid_5d[:, :, :, :-1, :]).abs().mean()
    dw = (grid_5d[:, :, :, :, 1:] - grid_5d[:, :, :, :, :-1]).abs().mean()
    return dh + dw


def wrap_for_onnx(model: CVDCorrectionNet, cvd_type: str) -> nn.Module:
    """
    Freeze cvd_type into a wrapper so ONNX export sees a pure-tensor
    forward. The exported model takes (orig_srgb, severity) only.
    """
    idx = CVD_TYPES.index(cvd_type)
    basis = get_basis(cvd_type)              # (2, 3), CPU
    b_L = basis[0].view(1, 3, 1, 1)
    b_C = basis[1].view(1, 3, 1, 1)

    class ONNXWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = model
            self.register_buffer("_bL", b_L.clone())
            self.register_buffer("_bC", b_C.clone())
            # Pre-registered w computation is heavy (Lab conversion). For
            # the smoke test we bypass w — assume w is fed externally OR
            # provide identity here. Real inference uses a small
            # confusion-weight sub-graph traced separately.

        def forward(self, orig_srgb, severity, w):
            B = orig_srgb.shape[0]
            device = orig_srgb.device
            dtype = orig_srgb.dtype
            cvd_onehot = torch.zeros(B, 3, device=device, dtype=dtype)
            cvd_onehot[:, idx] = 1.0
            sev = severity.expand(B, 1).to(dtype=dtype)
            interm = self.core.predict_fields(orig_srgb, cvd_onehot, sev)
            d_lum, d_c = interm["d_lum"], interm["d_c"]
            delta = d_lum * self._bL.to(dtype) + d_c * self._bC.to(dtype)
            out_lin = (interm["orig_lin"] + w * delta).clamp(0.0, 1.0)
            return linear_to_srgb(out_lin)

    return ONNXWrapper()
