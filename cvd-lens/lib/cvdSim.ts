/**
 * cvdSim — client-side CVD (dichromat) simulation, pure functions only.
 *
 * Extracted from the reverted cvdEngine.ts (6a6fc3c) Brettel path. NO
 * onnxruntime-web import here, so importing this pulls no wasm/ort code into the
 * bundle — it's a plain color transform used by the "CVD 시뮬레이션 보기" view.
 * Brettel 1997 matrices operate on LINEAR RGB; values match
 * cvdlens_v2/simulation.py::_BRETTEL_RGB.
 */
export type CVDType = "p" | "d" | "t";

// linear-RGB → linear-RGB Brettel matrices (row-major, LMS2RGB·MAT·RGB2LMS).
const BRETTEL: Record<CVDType, number[]> = {
  p: [0.11238179, 0.88761091, -0.00002372, 0.11238284, 0.88761824, 0.00000302, 0.00400747, -0.00400745, 1.0],
  d: [0.2927506, 0.70725077, -0.00001742, 0.29274943, 0.70724988, 0.00000721, -0.02233333, 0.02233337, 0.9999994],
  t: [0.49329609, 0.50670952, -0.00001212, 0.49321848, 0.5067755, 0.00001177, -3.01063704, 3.01067281, 0.99992812],
};

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
function srgbToLinear(c: number): number {
  return c > 0.04045 ? Math.pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
}
function linearToSrgb(c: number): number {
  const v = c > 0.0031308 ? 1.055 * Math.pow(Math.max(c, 0), 1 / 2.4) - 0.055 : 12.92 * c;
  return clamp01(v);
}

/** Brettel dichromat simulation at full severity. Returns a new ImageData. */
export function simulate(src: ImageData, type: CVDType): ImageData {
  const M = BRETTEL[type];
  const out = new Uint8ClampedArray(src.data.length);
  for (let i = 0; i < src.data.length; i += 4) {
    const r = srgbToLinear(src.data[i] / 255);
    const g = srgbToLinear(src.data[i + 1] / 255);
    const b = srgbToLinear(src.data[i + 2] / 255);
    const lr = M[0] * r + M[1] * g + M[2] * b;
    const lg = M[3] * r + M[4] * g + M[5] * b;
    const lb = M[6] * r + M[7] * g + M[8] * b;
    out[i] = Math.round(linearToSrgb(clamp01(lr)) * 255);
    out[i + 1] = Math.round(linearToSrgb(clamp01(lg)) * 255);
    out[i + 2] = Math.round(linearToSrgb(clamp01(lb)) * 255);
    out[i + 3] = src.data[i + 3];
  }
  return new ImageData(out, src.width, src.height);
}
