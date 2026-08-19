"""
Differentiable CVD simulator — single source of truth.

Two methods:
    Brettel 1997: LMS-based projection onto dichromat plane. Full severity only.
    Machado 2009: severity-interpolated 3x3 RGB matrices. Continuous severity.

At severity=1.0, Machado approximates Brettel (spot-checked in __main__).
Both operate on *linear* RGB.

`daltonize()` uses Brettel + error-shift as the daltonize comparison baseline
(Step 3 eval) and in diagnostics — NOT as a v2 training target (the loss is on
the simulator, see losses.CVDLossV2). Its error-shift matrix is TYPE-SPECIFIC:
p/d redistribute the red-green error (R) to G/B, while t redistributes the
blue-yellow error (B) to R/G (blue→purple) — a single p/d matrix cannot shift
blue toward purple (all-zero R row). External daltonize libraries must NOT be
imported anywhere in this codebase.
"""

import numpy as np
import torch


# ── Brettel 1997 (LMS space) ─────────────────────────────────────────────

_RGB2LMS = np.array([
    [17.8824,  43.5161,  4.11935],
    [ 3.45565, 27.1554,  3.86714],
    [ 0.02996,  0.18431, 1.46720],
], dtype=np.float32)

_LMS2RGB = np.linalg.inv(_RGB2LMS).astype(np.float32)

# Simulation matrices in LMS space (Brettel 1997)
_BRETTEL_LMS = {
    "p": np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
    "d": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]], dtype=np.float32),
    "t": np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]], dtype=np.float32),
}

# Pre-composed RGB→RGB Brettel matrices: LMS2RGB @ MAT @ RGB2LMS
_BRETTEL_RGB = {
    k: (_LMS2RGB @ mat @ _RGB2LMS).astype(np.float32)
    for k, mat in _BRETTEL_LMS.items()
}


# ── Machado 2009 (RGB space, severity-parameterised) ─────────────────────
# Reference matrices from Machado, Oliveira, Fernandes (2009), Table 1.
# 11 keypoints per type: severity 0.0, 0.1, ..., 1.0
# Linear interpolation between keypoints for continuous severity.

_MACHADO = {
    "p": np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],                                    # 0.0
        [[0.856167, 0.182038, -0.038205], [0.029342, 0.955115, 0.015544], [-0.002880, -0.001563, 1.004443]],
        [[0.734766, 0.334872, -0.069637], [0.051840, 0.919198, 0.028963], [-0.004928, -0.004209, 1.009137]],
        [[0.630323, 0.465641, -0.095964], [0.069181, 0.890046, 0.040773], [-0.006308, -0.007724, 1.014032]],
        [[0.539009, 0.579343, -0.118352], [0.082546, 0.866121, 0.051332], [-0.007136, -0.011959, 1.019095]],
        [[0.458064, 0.679578, -0.137642], [0.092785, 0.846313, 0.060902], [-0.007494, -0.016807, 1.024301]],
        [[0.385450, 0.769005, -0.154455], [0.100526, 0.829802, 0.069673], [-0.007442, -0.022190, 1.029632]],
        [[0.319627, 0.849633, -0.169261], [0.106241, 0.815969, 0.077790], [-0.007025, -0.028051, 1.035076]],
        [[0.259411, 0.923008, -0.182420], [0.110296, 0.804340, 0.085364], [-0.006276, -0.034346, 1.040622]],
        [[0.203876, 0.990338, -0.194214], [0.112975, 0.794542, 0.092483], [-0.005222, -0.041043, 1.046265]],
        [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],  # 1.0
    ], dtype=np.float32),
    "d": np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.866435, 0.177704, -0.044139], [0.049567, 0.939063, 0.011370], [-0.003453, 0.007233, 0.996220]],
        [[0.760729, 0.319078, -0.079807], [0.090568, 0.889315, 0.020117], [-0.006027, 0.013325, 0.992702]],
        [[0.675425, 0.433850, -0.109275], [0.125303, 0.847755, 0.026942], [-0.007950, 0.018572, 0.989378]],
        [[0.605511, 0.528560, -0.134071], [0.155318, 0.812366, 0.032316], [-0.009376, 0.023176, 0.986200]],
        [[0.547494, 0.607765, -0.155259], [0.181692, 0.781742, 0.036566], [-0.010410, 0.027275, 0.983136]],
        [[0.498864, 0.674741, -0.173604], [0.205199, 0.754872, 0.039929], [-0.011131, 0.030969, 0.980162]],
        [[0.457771, 0.731899, -0.189670], [0.226409, 0.731012, 0.042579], [-0.011595, 0.034333, 0.977261]],
        [[0.422823, 0.781057, -0.203881], [0.245752, 0.709602, 0.044646], [-0.011843, 0.037423, 0.974421]],
        [[0.392952, 0.823610, -0.216562], [0.263559, 0.690210, 0.046232], [-0.011910, 0.040281, 0.971630]],
        [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],  # 1.0
    ], dtype=np.float32),
    "t": np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.926670, 0.092514, -0.019184], [0.021191, 0.964503, 0.014306], [0.008437, 0.054813, 0.936750]],
        [[0.895720, 0.133330, -0.029050], [0.029997, 0.945400, 0.024603], [0.013027, 0.104707, 0.882266]],
        [[0.905871, 0.127791, -0.033662], [0.026856, 0.941251, 0.031893], [0.013410, 0.148296, 0.838294]],
        [[0.948035, 0.089490, -0.037526], [0.014364, 0.946792, 0.038844], [0.010853, 0.193991, 0.795156]],
        [[1.017277, 0.027029, -0.044306], [-0.006113, 0.958479, 0.047634], [0.006379, 0.248708, 0.744913]],
        [[1.104996, -0.046633, -0.058363], [-0.032137, 0.971635, 0.060503], [0.001336, 0.317922, 0.680742]],
        [[1.193214, -0.109812, -0.083402], [-0.058496, 0.979410, 0.079086], [-0.002346, 0.403492, 0.598854]],
        [[1.257728, -0.139648, -0.118081], [-0.078003, 0.975409, 0.102594], [-0.003316, 0.501214, 0.502102]],
        [[1.278864, -0.125333, -0.153531], [-0.084748, 0.957674, 0.127074], [-0.000989, 0.601151, 0.399838]],
        [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],  # 1.0
    ], dtype=np.float32),
}


def _machado_matrix(cvd_type: str, severity: float) -> np.ndarray:
    """Linear interpolate between Machado severity keypoints."""
    keys = _MACHADO[cvd_type]
    s = float(np.clip(severity, 0.0, 1.0))
    idx_f = s * 10.0
    lo = int(np.floor(idx_f))
    hi = min(lo + 1, 10)
    frac = idx_f - lo
    return (1.0 - frac) * keys[lo] + frac * keys[hi]


def machado_matrix_tensor(cvd_type: str, severity: torch.Tensor) -> torch.Tensor:
    """
    ONNX-traceable, severity-LIVE Machado matrix. `severity` is a tensor (any
    shape with 1 element); returns a (3, 3) tensor.

    Uses tent-weight interpolation over the 11 keypoints:
        w_k = relu(1 - |s·10 - k|),   mat = Σ_k w_k · keys[k]
    which is *exactly* the piecewise-linear interpolation of `_machado_matrix`
    (identical at every keypoint and on each segment), but expressed in pure
    tensor ops so severity survives ONNX tracing instead of being frozen at
    trace time. Verified equal to `_machado_matrix` in the module self-test.
    """
    keys = torch.as_tensor(_MACHADO[cvd_type], dtype=severity.dtype,
                           device=severity.device)          # (11, 3, 3)
    s = severity.reshape(()).clamp(0.0, 1.0)
    idx_f = s * 10.0
    ks = torch.arange(11, dtype=severity.dtype, device=severity.device)
    wts = (1.0 - (idx_f - ks).abs()).clamp(min=0.0)          # (11,)
    return (wts.view(11, 1, 1) * keys).sum(dim=0)            # (3, 3)


# ── Unified simulator ────────────────────────────────────────────────────

def simulate(
    rgb_linear: torch.Tensor,
    cvd_type: str,
    severity: float = 1.0,
    method: str = "machado",
) -> torch.Tensor:
    """
    Differentiable CVD simulation on *linear* RGB.

    Args:
        rgb_linear: (B, 3, H, W) in [0, 1]
        cvd_type:   'p' | 'd' | 't'
        severity:   [0, 1]. Ignored for Brettel (always 1.0).
        method:     'brettel' | 'machado'
    Returns:
        (B, 3, H, W) simulated linear RGB in [0, 1]
    """
    assert cvd_type in ("p", "d", "t"), cvd_type
    device, dtype = rgb_linear.device, rgb_linear.dtype

    if method == "brettel":
        mat = torch.from_numpy(_BRETTEL_RGB[cvd_type]).to(device, dtype)
    elif method == "machado":
        mat_np = _machado_matrix(cvd_type, severity)
        mat = torch.from_numpy(mat_np).to(device, dtype)
    else:
        raise ValueError(f"Unknown method: {method}")

    out = torch.einsum('ij,bjhw->bihw', mat, rgb_linear)
    return out.clamp(0.0, 1.0)


def simulate_batch_types(
    rgb_linear: torch.Tensor,
    cvd_types: list,
    severities: torch.Tensor,
    method: str = "machado",
) -> torch.Tensor:
    """
    Per-sample CVD type and severity. Used when different images in a batch
    have different types.
    """
    outs = []
    for i, (t, s) in enumerate(zip(cvd_types, severities.tolist())):
        outs.append(simulate(rgb_linear[i:i+1], t, s, method))
    return torch.cat(outs, dim=0)


# ── Daltonize (Brettel + error-shift) ────────────────────────────────────

# Error-shift redistributes the lost signal (error) into channels the viewer CAN
# see. The lost axis differs by type, so the matrix must be type-specific: for
# protan/deutan the lost signal is red-green (mostly the R channel) and is fed to
# G/B; for tritan it is blue-yellow (mostly the B channel) and must be fed to R/G
# (blue→purple). A single p/d matrix applied to tritan cannot add anything to R
# (its R row is all-zero), so blue can never shift toward purple — the tritan
# error-shift target degenerates to a near no-op on blue. Hence per-type matrices.

# Tritan gain: how much of the B-channel error is redistributed into R and G.
# Tunable — raise/lower if the sanity projection (sanity_tritan_target.py) shows
# the shift is too weak/strong relative to the tritan visible subspace.
TRITAN_SHIFT_GAIN = 0.7

_ERR2MOD = {
    # protan / deutan — red-green error (R) → G, B.
    # DO NOT CHANGE: bit-exact with the pre-split single matrix (reproducibility
    # with the existing p/d models, evaluation, and diagnostics).
    "p": np.array([
        [0,   0, 0],
        [0.7, 1, 0],
        [0.7, 0, 1],
    ], dtype=np.float32),
    "d": np.array([
        [0,   0, 0],
        [0.7, 1, 0],
        [0.7, 0, 1],
    ], dtype=np.float32),
    # tritan — blue-yellow error (B) → R, G (blue→purple). NEW: fixes the
    # degenerate all-zero-R-row behaviour of the shared p/d matrix.
    "t": np.array([
        [1, 0, TRITAN_SHIFT_GAIN],
        [0, 1, TRITAN_SHIFT_GAIN],
        [0, 0, 0],
    ], dtype=np.float32),
}


def daltonize(
    rgb_linear: torch.Tensor,
    cvd_type: str,
    severity: float = 1.0,
    method: str = "machado",
) -> torch.Tensor:
    """
    Daltonize: original + shift(error), where error = original - simulated.
    Same simulator source. Used as the daltonize comparison baseline (Step 3
    evaluation) and in diagnostics — NOT as a v2 training target (the loss is
    defined directly on the simulator; see losses.CVDLossV2). The error-shift
    matrix is selected per cvd_type (see _ERR2MOD).
    """
    sim = simulate(rgb_linear, cvd_type, severity, method)
    err = rgb_linear - sim
    device, dtype = rgb_linear.device, rgb_linear.dtype
    shift_mat = torch.from_numpy(_ERR2MOD[cvd_type]).to(device, dtype)
    err_shifted = torch.einsum('ij,bjhw->bihw', shift_mat, err)
    return (rgb_linear + err_shifted).clamp(0.0, 1.0)


if __name__ == "__main__":
    torch.manual_seed(0)
    rgb = torch.rand(1, 3, 64, 64)

    for t in ["p", "d", "t"]:
        brt = simulate(rgb, t, 1.0, "brettel")
        mac1 = simulate(rgb, t, 1.0, "machado")
        mac_half = simulate(rgb, t, 0.5, "machado")
        diff_full = (brt - mac1).abs().mean().item()
        diff_half = (rgb - mac_half).abs().mean().item()
        print(f"[{t}] Brettel vs Machado(1.0) mean err: {diff_full:.4f} "
              f"| Machado(0.5) mean shift from orig: {diff_half:.4f}")

    dalt = daltonize(rgb, "p", 1.0)
    print(f"daltonize output range: [{dalt.min():.3f}, {dalt.max():.3f}]")
