/**
 * cvdOnnx — client-side ONNX Runtime Web 추론 (p/d 학습 모델을 브라우저에서 실행).
 * 서버 /infer와 동일한 학습 모델(cvdlens_{p,d}.onnx)을 사용자 기기에서 추론.
 * 그래프 IO: srgb (1,3,256,256) f32 + severity (1,1) f32 → out_srgb (1,3,256,256).
 * 현재 배포 그래프의 출력은 f64이며 호출측에는 f32로 정규화해 반환한다.
 */
import type * as Ort from "onnxruntime-web/wasm";
import type { CVDType } from "./cvd";

type ModelCVDType = Exclude<CVDType, "t">;
type Runtime = typeof import("onnxruntime-web/wasm");

// WASM 전용 번들과 같은 빌드의 실행 파일을 사용한다. 기본 번들은 별도의
// jsep 실행 파일을 요구하므로 executionProviders: ["wasm"]만으로는 충분하지 않다.
// npm predev/prebuild가 public/ort를 설치된 패키지와 동기화한다.
// useEffect에서 필요할 때만 가져와 서버 사전 렌더링에서 WASM을 초기화하지 않는다.
let runtimePromise: Promise<Runtime> | undefined;
function getRuntime(): Promise<Runtime> {
  const pending = runtimePromise ?? import("onnxruntime-web/wasm").then((ort) => {
    if (typeof window !== "undefined") {
      const version = encodeURIComponent(ort.env.versions.web ?? ort.env.versions.common);
      const assetUrl = (extension: string) =>
        new URL(`/ort/ort-wasm-simd-threaded.${extension}?v=${version}`, window.location.origin).href;
      ort.env.wasm.wasmPaths = { mjs: assetUrl("mjs"), wasm: assetUrl("wasm") };
    }
    // 스레드 off = SharedArrayBuffer/교차출처격리 불필요.
    ort.env.wasm.numThreads = 1;
    return ort;
  });
  runtimePromise = pending;
  pending.catch(() => { if (runtimePromise === pending) runtimePromise = undefined; });
  return pending;
}

export const ONNX_SIZE = 256;

const sessions: Partial<Record<ModelCVDType, Promise<Ort.InferenceSession>>> = {};

function getSession(type: ModelCVDType): Promise<Ort.InferenceSession> {
  const pending = sessions[type] ?? getRuntime().then((ort) => ort.InferenceSession.create(`/models/cvdlens_${type}.onnx`, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  }));
  sessions[type] = pending;
  pending.catch(() => { if (sessions[type] === pending) delete sessions[type]; });
  return pending;
}

/** 세션 미리 로드(첫 프레임 지연 감소). */
export function preloadSession(type: ModelCVDType): Promise<unknown> {
  return getSession(type);
}

/**
 * 256×256 입력(ImageData)을 학습 모델로 보정. 반환: CHW Float32Array [3*256*256], 값 [0,1].
 * (호출측에서 delta = out - in 계산 후 원해상도로 업샘플·합성)
 */
export async function runOnnxCorrection(
  type: ModelCVDType,
  src256: ImageData,
  severity: number,
): Promise<Float32Array> {
  const [ort, session] = await Promise.all([getRuntime(), getSession(type)]);
  const N = ONNX_SIZE * ONNX_SIZE;
  const chw = new Float32Array(3 * N);
  const d = src256.data;
  for (let i = 0; i < N; i++) {
    chw[i] = d[i * 4] / 255;         // R plane
    chw[N + i] = d[i * 4 + 1] / 255; // G plane
    chw[2 * N + i] = d[i * 4 + 2] / 255; // B plane
  }
  const feeds: Record<string, Ort.Tensor> = {
    srgb: new ort.Tensor("float32", chw, [1, 3, ONNX_SIZE, ONNX_SIZE]),
    severity: new ort.Tensor("float32", new Float32Array([severity]), [1, 1]),
  };
  const results = await session.run(feeds);
  const output = results.out_srgb.data;
  if (output instanceof Float32Array) return output;
  if (output instanceof Float64Array) return Float32Array.from(output);
  throw new Error(`지원하지 않는 모델 출력 형식: ${results.out_srgb.type}`);
}
