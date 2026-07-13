"""
Visible subspace basis for each CVD type.

For a given CVD type, the simulator projects 3D RGB onto a 2D subspace
(the dichromat's visible plane). Any RGB delta that lies along the
confusion line (perpendicular to that plane) is invisible to the CVD viewer.

To force the model to only produce visible changes, we parametrize:

    delta = d_lum * b_L + d_c * b_C

where {b_L, b_C} span the visible subspace and are orthogonal to the
confusion direction. Guarantees:
    dot(delta, confusion_dir) == 0

Derivation (linear RGB):
    Confusion direction = 3D null direction of the Brettel simulator I - S,
    where S = simulate matrix (I - S maps to what CVD viewer LOSES).
    Visible subspace = orthogonal complement, span of S's image.

We compute this once at import from Brettel matrices and register as
buffers on demand.
"""

import numpy as np
import torch

from .simulation import _BRETTEL_RGB


def _visible_basis(cvd_type: str) -> np.ndarray:
    """
    Return (2, 3) matrix whose rows span the visible subspace for cvd_type
    in linear RGB. Rows are orthonormal:
        row 0 = b_L (aligned toward luminance)
        row 1 = b_C (color contrast direction, orthogonal to b_L and confusion)
    """
    S = _BRETTEL_RGB[cvd_type]                 # simulator matrix
    # Confusion direction = eigenvector of S with eigenvalue 0
    # (equivalently: null space of S). Use SVD for numerical stability.
    U, sig, Vt = np.linalg.svd(S)
    # Smallest singular value's right vector is the confusion direction
    order = np.argsort(sig)
    conf_dir = Vt[order[0]]                    # (3,)
    conf_dir = conf_dir / np.linalg.norm(conf_dir)

    # Visible plane: orthogonal complement of conf_dir in R^3
    # Get any 2 orthonormal vectors orthogonal to conf_dir
    # Prefer b_L aligned with luminance direction
    lum = np.array([0.2126729, 0.7151522, 0.0721750], dtype=np.float32)  # BT.709 linear luma
    # Project lum onto visible plane
    b_L = lum - np.dot(lum, conf_dir) * conf_dir
    b_L = b_L / (np.linalg.norm(b_L) + 1e-8)
    # b_C = orthogonal to both b_L and conf_dir
    b_C = np.cross(conf_dir, b_L)
    b_C = b_C / (np.linalg.norm(b_C) + 1e-8)

    basis = np.stack([b_L, b_C], axis=0).astype(np.float32)   # (2, 3)
    return basis


# Precompute for all CVD types
_BASIS = {t: _visible_basis(t) for t in ["p", "d", "t"]}
_CONF_DIR = {}
for t in ["p", "d", "t"]:
    S = _BRETTEL_RGB[t]
    _, sig, Vt = np.linalg.svd(S)
    order = np.argsort(sig)
    d = Vt[order[0]].astype(np.float32)
    _CONF_DIR[t] = d / np.linalg.norm(d)


def get_basis(cvd_type: str, device=None, dtype=None) -> torch.Tensor:
    """Return (2, 3) basis tensor: rows are [b_L, b_C]."""
    t = torch.from_numpy(_BASIS[cvd_type])
    if device is not None or dtype is not None:
        t = t.to(device=device, dtype=dtype)
    return t


def get_confusion_dir(cvd_type: str, device=None, dtype=None) -> torch.Tensor:
    """Return (3,) unit vector in confusion direction."""
    t = torch.from_numpy(_CONF_DIR[cvd_type])
    if device is not None or dtype is not None:
        t = t.to(device=device, dtype=dtype)
    return t


def compose_delta(
    d_lum: torch.Tensor,
    d_c: torch.Tensor,
    cvd_type: str,
) -> torch.Tensor:
    """
    Compose per-pixel RGB delta from scalar fields.

    Args:
        d_lum: (B, 1, H, W) or (B, H, W) — luminance scalar field
        d_c:   (B, 1, H, W) or (B, H, W) — color-contrast scalar field
        cvd_type: 'p' | 'd' | 't'
    Returns:
        delta: (B, 3, H, W) linear RGB delta. Lies in visible subspace.
    """
    if d_lum.dim() == 3:
        d_lum = d_lum.unsqueeze(1)
    if d_c.dim() == 3:
        d_c = d_c.unsqueeze(1)
    basis = get_basis(cvd_type, device=d_lum.device, dtype=d_lum.dtype)  # (2, 3)
    b_L = basis[0].view(1, 3, 1, 1)
    b_C = basis[1].view(1, 3, 1, 1)
    return d_lum * b_L + d_c * b_C


if __name__ == "__main__":
    for t in ["p", "d", "t"]:
        basis = get_basis(t)
        conf = get_confusion_dir(t)
        # Orthogonality checks
        dot_LC = torch.dot(basis[0], basis[1]).item()
        dot_Lconf = torch.dot(basis[0], conf).item()
        dot_Cconf = torch.dot(basis[1], conf).item()
        norm_L = torch.linalg.norm(basis[0]).item()
        norm_C = torch.linalg.norm(basis[1]).item()
        print(f"[{t}]")
        print(f"  b_L: {basis[0].tolist()}  norm={norm_L:.4f}")
        print(f"  b_C: {basis[1].tolist()}  norm={norm_C:.4f}")
        print(f"  conf_dir: {conf.tolist()}")
        print(f"  <b_L, b_C> = {dot_LC:.6f}")
        print(f"  <b_L, conf> = {dot_Lconf:.6f}")
        print(f"  <b_C, conf> = {dot_Cconf:.6f}")

    # compose_delta check: confusion component must be zero
    d_lum = torch.randn(1, 1, 8, 8)
    d_c = torch.randn(1, 1, 8, 8)
    for t in ["p", "d", "t"]:
        delta = compose_delta(d_lum, d_c, t)
        conf = get_confusion_dir(t).view(1, 3, 1, 1)
        conf_component = (delta * conf).sum(dim=1)
        print(f"[{t}] max |delta . conf_dir| = {conf_component.abs().max().item():.2e}")
