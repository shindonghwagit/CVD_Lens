"""
Phase 3 - ONNX 변환 + float16 양자화

사용법:
    py export_onnx.py --ckpt C:/Users/SCH/Downloads/best-epoch=026-val_loss=0.0265.ckpt
    py export_onnx.py --ckpt C:/Users/SCH/Downloads/best-epoch=026-val_loss=0.0265.ckpt --no-fp16
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import CVDLitModule


def export(ckpt_path: str, output_dir: str, fp16: bool = True):
    ckpt_path = Path(ckpt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 체크포인트 로드 ──────────────────────────────
    print(f"체크포인트 로드: {ckpt_path}")
    module = CVDLitModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    module.eval()
    net = module.model  # smp.Unet

    # ── 2. fp32 ONNX 내보내기 ──────────────────────────
    dummy = torch.zeros(1, 4, 256, 256)
    fp32_path = output_dir / "cvdlens_fp32.onnx"

    print(f"ONNX(fp32) 내보내기: {fp32_path}")
    torch.onnx.export(
        net,
        dummy,
        str(fp32_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # 구 exporter 사용 (Windows cp949 호환)
    )
    size_fp32 = fp32_path.stat().st_size / 1024 / 1024
    print(f"  → {size_fp32:.1f} MB")

    # ── 3. ONNX 검증 ──────────────────────────────────
    try:
        import onnx
        model_onnx = onnx.load(str(fp32_path))
        onnx.checker.check_model(model_onnx)
        print("  ONNX 검증 통과")
    except ImportError:
        print("  (onnx 패키지 없음 - 검증 스킵. pip install onnx 권장)")

    # ── 4. 추론 결과 비교 (PyTorch vs ONNX Runtime) ───
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        test_input = torch.rand(1, 4, 256, 256)
        with torch.no_grad():
            pt_out = net(test_input).numpy()
        ort_out = sess.run(["output"], {"input": test_input.numpy()})[0]
        max_diff = np.abs(pt_out - ort_out).max()
        print(f"  PyTorch vs ONNX 최대 오차: {max_diff:.6f}")
        assert max_diff < 1e-4, f"오차 너무 큼: {max_diff}"
        print("  추론 일치 검증 통과")
    except ImportError:
        print("  (onnxruntime 없음 - 추론 검증 스킵. pip install onnxruntime 권장)")

    # ── 5. float16 양자화 ─────────────────────────────
    if fp16:
        try:
            from onnxconverter_common import float16 as f16_converter
            fp16_path = output_dir / "cvdlens_fp16.onnx"
            print(f"\nfloat16 변환 중: {fp16_path}")
            model_fp32 = onnx.load(str(fp32_path))
            model_fp16 = f16_converter.convert_float_to_float16(
                model_fp32,
                keep_io_types=True,   # 입출력은 fp32 유지 (JS 호환성)
            )
            import onnx
            onnx.save(model_fp16, str(fp16_path))
            size_fp16 = fp16_path.stat().st_size / 1024 / 1024
            print(f"  → {size_fp16:.1f} MB  (압축률: {size_fp32/size_fp16:.1f}x)")
        except ImportError:
            print("  (onnxconverter-common 없음 - fp16 스킵)")
            print("  pip install onnxconverter-common")
            fp16 = False

    # ── 6. 결과 요약 ──────────────────────────────────
    print("\n" + "="*50)
    print("변환 완료")
    print(f"  fp32: {fp32_path}  ({size_fp32:.1f} MB)")
    if fp16:
        print(f"  fp16: {output_dir / 'cvdlens_fp16.onnx'}  ({size_fp16:.1f} MB)")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVDLens ONNX 변환")
    parser.add_argument("--ckpt",       required=True,              help="체크포인트 경로 (.ckpt)")
    parser.add_argument("--output-dir", default="./outputs/onnx",   help="출력 디렉토리 (기본: ./outputs/onnx)")
    parser.add_argument("--no-fp16",    action="store_true",        help="float16 변환 생략")
    args = parser.parse_args()

    export(
        ckpt_path=args.ckpt,
        output_dir=args.output_dir,
        fp16=not args.no_fp16,
    )
