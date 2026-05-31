"""
Export Phase 2 mask predictor to ONNX.

Usage:
    python export_mask_onnx.py --ckpt outputs/checkpoints_mask/last.ckpt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model.mask_lit_module import MaskLitModule


def export(ckpt_path: str, output_dir: str = "./outputs/onnx_mask") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    module = MaskLitModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    module.eval()
    export_model = torch.nn.Sequential(module.model, torch.nn.Sigmoid()).eval()

    dummy = torch.randn(1, 8, 256, 256)
    out_path = output_dir / "cvdlens_mask_fp32.onnx"

    torch.onnx.export(
        export_model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["mask"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "mask": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Saved ONNX: {out_path}")

    model_onnx = onnx.load(str(out_path))
    onnx.checker.check_model(model_onnx)

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    out = sess.run(["mask"], {"input": dummy.numpy()})[0]

    assert out.shape == (1, 1, 256, 256), f"unexpected output shape: {out.shape}"
    assert np.isfinite(out).all(), "ONNX output contains NaN/Inf"
    assert 0.0 <= out.min() and out.max() <= 1.0, f"output range error: [{out.min():.3f}, {out.max():.3f}]"

    print(f"ONNX check passed: shape={out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")
    print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/onnx_mask")
    args = parser.parse_args()
    export(args.ckpt, args.output_dir)
