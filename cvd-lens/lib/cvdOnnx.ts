/**
 * cvdOnnx — client-side ONNX Runtime Web 추론 (p/d 학습 모델을 브라우저에서 실행).
 * 서버 /infer와 동일한 학습 모델(cvdlens_{p,d}.onnx)을 사용자 기기에서 추론.
 * 그래프 IO: srgb (1,3,256,256) f32 + severity (1,1) f32 → out_srgb (1,3,256,256) f32.
 */
import * as ort from "onnxruntime-web";
import type { CVDType } from "./cvdSim";

// wasm 바이너리는 CDN에서 (번들러 wasm 처리 회피). 스레드 off = SharedArrayBuffer/교차출처격리 불필요.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
ort.env.wasm.numThreads = 1;

export const ONNX_SIZE = 256;

const sessions: Partial<Record<CVDType, Promise<ort.InferenceSession>>> = {};

function getSession(type: CVDType): Promise<ort.InferenceSession> {
  return (sessions[type] ??= ort.InferenceSession.create(`/models/cvdlens_${type}.onnx`, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  }));
}

/** 세션 미리 로드(첫 프레임 지연 감소). */
export function preloadSession(type: CVDType): Promise<unknown> {
  return getSession(type);
}

/**
 * 256×256 입력(ImageData)을 학습 모델로 보정. 반환: CHW Float32Array [3*256*256], 값 [0,1].
 * (호출측에서 delta = out - in 계산 후 원해상도로 업샘플·합성)
 */
export async function runOnnxCorrection(
  type: CVDType,
  src256: ImageData,
  severity: number,
): Promise<Float32Array> {
  const session = await getSession(type);
  const N = ONNX_SIZE * ONNX_SIZE;
  const chw = new Float32Array(3 * N);
  const d = src256.data;
  for (let i = 0; i < N; i++) {
    chw[i] = d[i * 4] / 255;         // R plane
    chw[N + i] = d[i * 4 + 1] / 255; // G plane
    chw[2 * N + i] = d[i * 4 + 2] / 255; // B plane
  }
  const feeds: Record<string, ort.Tensor> = {
    srgb: new ort.Tensor("float32", chw, [1, 3, ONNX_SIZE, ONNX_SIZE]),
    severity: new ort.Tensor("float32", new Float32Array([severity]), [1, 1]),
  };
  const results = await session.run(feeds);
  return results.out_srgb.data as Float32Array;
}
