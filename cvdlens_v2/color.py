"""
Differentiable color space conversions.

All operations work on (B, 3, H, W) tensors in [0, 1] and are
autograd-friendly. Pow operations are stabilized against near-zero
values for fp16 safety.
"""

import torch


# ── sRGB ↔ linear RGB ────────────────────────────────────────────────────

def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) sRGB [0,1] → linear RGB [0,1]."""
    mask = rgb > 0.04045
    return torch.where(
        mask,
        ((rgb.clamp(min=0.0) + 0.055) / 1.055).clamp(min=1e-8).pow(2.4),
        rgb / 12.92,
    )


def linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) linear RGB [0,1] → sRGB [0,1]."""
    mask = rgb > 0.0031308
    return torch.where(
        mask,
        1.055 * rgb.clamp(min=1e-8).pow(1.0 / 2.4) - 0.055,
        12.92 * rgb,
    ).clamp(0.0, 1.0)


# ── linear RGB → XYZ → Lab (D65) ─────────────────────────────────────────

_M_RGB2XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float32)

_M_XYZ2RGB = torch.tensor([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=torch.float32)

_D65_WHITE = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)

_DELTA = 6.0 / 29.0


def _xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    white = _D65_WHITE.to(xyz.device, xyz.dtype).view(1, 3, 1, 1)
    xyz_n = xyz / white
    f = torch.where(
        xyz_n > _DELTA ** 3,
        xyz_n.clamp(min=1e-8).pow(1.0 / 3.0),
        xyz_n / (3 * _DELTA ** 2) + 4.0 / 29.0,
    )
    L = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([L, a, b], dim=1)


def _lab_to_xyz(lab: torch.Tensor) -> torch.Tensor:
    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    f = torch.cat([fx, fy, fz], dim=1)
    xyz_n = torch.where(
        f > _DELTA,
        f.pow(3.0),
        3.0 * _DELTA ** 2 * (f - 4.0 / 29.0),
    )
    white = _D65_WHITE.to(lab.device, lab.dtype).view(1, 3, 1, 1)
    return xyz_n * white


def rgb_to_lab(rgb_linear: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) linear RGB [0,1] → CIELAB. Assumes input is *linear* RGB."""
    M = _M_RGB2XYZ.to(rgb_linear.device, rgb_linear.dtype)
    xyz = torch.einsum('ij,bjhw->bihw', M, rgb_linear)
    return _xyz_to_lab(xyz)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) CIELAB → linear RGB [0,1]."""
    xyz = _lab_to_xyz(lab)
    M = _M_XYZ2RGB.to(lab.device, lab.dtype)
    rgb = torch.einsum('ij,bjhw->bihw', M, xyz)
    return rgb.clamp(0.0, 1.0)


def delta_e_lab(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """Euclidean CIE76 delta-E. Input/output shape preserved on channel dim."""
    return torch.sqrt(((lab1 - lab2) ** 2).sum(dim=1, keepdim=True) + 1e-6)


if __name__ == "__main__":
    torch.manual_seed(0)
    rgb = torch.rand(2, 3, 32, 32)
    lin = srgb_to_linear(rgb)
    srgb_back = linear_to_srgb(lin)
    print(f"sRGB round-trip max err: {(rgb - srgb_back).abs().max():.6f}")

    lab = rgb_to_lab(lin)
    lin_back = lab_to_rgb(lab)
    print(f"Lab round-trip max err:  {(lin - lin_back).abs().max():.6f}")
    print(f"L range: [{lab[:,0].min():.2f}, {lab[:,0].max():.2f}]")
    print(f"a range: [{lab[:,1].min():.2f}, {lab[:,1].max():.2f}]")
    print(f"b range: [{lab[:,2].min():.2f}, {lab[:,2].max():.2f}]")
