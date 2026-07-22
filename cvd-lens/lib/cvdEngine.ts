/**
 * cvdEngine — framework-independent, in-browser CVD correction via ONNX Runtime Web.
 *
 * Replaces the old server round-trip (POST /infer). Three per-type graphs
 * (cvdlens_{p,d,t}.onnx, step-9000 model_best) are lazy-loaded from
 * /public/models on first use. WASM binaries come from the jsDelivr CDN so
 * Vercel never has to serve .wasm (sidesteps MIME/size concerns).
 *
 * Public API:
 *   init(type)                       → load session, returns backend name
 *   correct(imageData, type, sev)    → corrected ImageData (letterboxed 256 → restored)
 *   simulate(imageData, type)        → Brettel CVD simulation (for the comparison view)
 *   lastBackend()                    → "webgpu" | "wasm"
 */
import * as ort from "onnxruntime-web";

export type CVDType = "p" | "d" | "t";

const SIZE = 256;
const ORT_VERSION = "1.27.0";

// WASM binaries from CDN (models stay local under /models).
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

let _backend = "";
const _sessions = new Map<CVDType, ort.InferenceSession>();
const _pending = new Map<CVDType, Promise<ort.InferenceSession>>();

export function lastBackend(): string {
  return _backend;
}

function preferredEPs(): string[] {
  const eps: string[] = [];
  // WebGPU first (no COOP/COEP needed), then WASM.
  if (typeof navigator !== "undefined" && (navigator as unknown as { gpu?: unknown }).gpu) {
    eps.push("webgpu");
  }
  eps.push("wasm");
  return eps;
}

async function createSession(type: CVDType): Promise<ort.InferenceSession> {
  // WASM threads only when cross-origin isolated (COOP/COEP); else SIMD single-thread.
  const isolated =
    typeof crossOriginIsolated !== "undefined" && crossOriginIsolated === true;
  ort.env.wasm.numThreads = isolated
    ? Math.min(4, (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 1)
    : 1;

  const url = `/models/cvdlens_${type}.onnx`;
  let lastErr: unknown;
  for (const ep of preferredEPs()) {
    try {
      const sess = await ort.InferenceSession.create(url, {
        executionProviders: [ep as ort.InferenceSession.ExecutionProviderConfig],
      });
      _backend = ep;
      return sess;
    } catch (e) {
      lastErr = e;
      // fall through to the next backend in the ladder
      console.warn(`[cvdEngine] executionProvider "${ep}" unavailable — falling back`, e);
    }
  }
  throw new Error(`cvdEngine: no execution provider available (${String(lastErr)})`);
}

export async function init(type: CVDType): Promise<string> {
  await getSession(type);
  return _backend;
}

async function getSession(type: CVDType): Promise<ort.InferenceSession> {
  const cached = _sessions.get(type);
  if (cached) return cached;
  let p = _pending.get(type);
  if (!p) {
    p = createSession(type);
    _pending.set(type, p);
  }
  const sess = await p;
  _sessions.set(type, sess);
  return sess;
}

// ── canvas helpers ──────────────────────────────────────────────────────
function makeCanvas(w: number, h: number): HTMLCanvasElement | OffscreenCanvas {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

function ctx2d(c: HTMLCanvasElement | OffscreenCanvas): CanvasRenderingContext2D {
  return c.getContext("2d") as unknown as CanvasRenderingContext2D;
}

type Box = { dx: number; dy: number; dw: number; dh: number };

function letterboxBox(w: number, h: number): Box {
  const scale = Math.min(SIZE / w, SIZE / h);
  const dw = Math.max(1, Math.round(w * scale));
  const dh = Math.max(1, Math.round(h * scale));
  return { dx: Math.floor((SIZE - dw) / 2), dy: Math.floor((SIZE - dh) / 2), dw, dh };
}

// ── tensor pack / unpack (pure, testable) ───────────────────────────────
/** RGBA uint8 (SIZE*SIZE*4) → planar CHW float32 [0,1], RGB only. */
export function rgbaToCHW(rgba: Uint8ClampedArray | Uint8Array): Float32Array {
  const n = SIZE * SIZE;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = rgba[i * 4] / 255; // R plane
    out[n + i] = rgba[i * 4 + 1] / 255; // G plane
    out[2 * n + i] = rgba[i * 4 + 2] / 255; // B plane
  }
  return out;
}

/** planar CHW float32 [0,1] → RGBA uint8 (SIZE*SIZE*4), alpha=255. */
export function chwToRGBA(chw: Float32Array): Uint8ClampedArray {
  const n = SIZE * SIZE;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = Math.round(clamp01(chw[i]) * 255);
    out[i * 4 + 1] = Math.round(clamp01(chw[n + i]) * 255);
    out[i * 4 + 2] = Math.round(clamp01(chw[2 * n + i]) * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** Low-level inference: CHW float32 (1,3,256,256) + severity → CHW float32 out. */
export async function runTensor(
  type: CVDType,
  chw: Float32Array,
  severity: number
): Promise<Float32Array> {
  const sess = await getSession(type);
  const feeds: Record<string, ort.Tensor> = {
    srgb: new ort.Tensor("float32", chw, [1, 3, SIZE, SIZE]),
    severity: new ort.Tensor("float32", Float32Array.from([severity]), [1, 1]),
  };
  const res = await sess.run(feeds);
  return res.out_srgb.data as Float32Array;
}

// ── public: correct ─────────────────────────────────────────────────────
export interface CorrectResult {
  image: ImageData;
  ms: number;
  backend: string;
}

/**
 * Correct `src` for `type` at `severity` (0..1). Internally letterboxes to
 * 256×256 (aspect preserved, black pad), runs the model, then crops the
 * content box back out and rescales to the original resolution.
 */
export async function correct(
  src: ImageData,
  type: CVDType,
  severity = 1.0
): Promise<CorrectResult> {
  const box = letterboxBox(src.width, src.height);

  // src ImageData → scaled onto a black 256 canvas
  const srcCanvas = makeCanvas(src.width, src.height);
  ctx2d(srcCanvas).putImageData(src, 0, 0);
  const inCanvas = makeCanvas(SIZE, SIZE);
  const ictx = ctx2d(inCanvas);
  ictx.fillStyle = "#000";
  ictx.fillRect(0, 0, SIZE, SIZE);
  ictx.drawImage(srcCanvas as CanvasImageSource, 0, 0, src.width, src.height, box.dx, box.dy, box.dw, box.dh);
  const inData = ictx.getImageData(0, 0, SIZE, SIZE);

  const chw = rgbaToCHW(inData.data);
  const t0 = now();
  const outChw = await runTensor(type, chw, severity);
  const ms = now() - t0;

  // out CHW → 256 RGBA → crop content box → rescale to original size
  const outRGBA = chwToRGBA(outChw);
  const outCanvas = makeCanvas(SIZE, SIZE);
  const octx = ctx2d(outCanvas);
  const outImageData = octx.createImageData(SIZE, SIZE);
  outImageData.data.set(outRGBA);
  octx.putImageData(outImageData, 0, 0);
  const resCanvas = makeCanvas(src.width, src.height);
  const rctx = ctx2d(resCanvas);
  rctx.drawImage(outCanvas as CanvasImageSource, box.dx, box.dy, box.dw, box.dh, 0, 0, src.width, src.height);

  return { image: rctx.getImageData(0, 0, src.width, src.height), ms, backend: _backend };
}

// ── public: simulate (Brettel 1997, JS) ─────────────────────────────────
// Pre-composed linear-RGB→linear-RGB Brettel matrices (LMS2RGB · MAT · RGB2LMS),
// exact values from cvdlens_v2/simulation.py::_BRETTEL_RGB.
const BRETTEL: Record<CVDType, number[]> = {
  p: [0.11238179, 0.88761091, -0.00002372, 0.11238284, 0.88761824, 0.00000302, 0.00400747, -0.00400745, 1.0],
  d: [0.2927506, 0.70725077, -0.00001742, 0.29274943, 0.70724988, 0.00000721, -0.02233333, 0.02233337, 0.9999994],
  t: [0.49329609, 0.50670952, -0.00001212, 0.49321848, 0.5067755, 0.00001177, -3.01063704, 3.01067281, 0.99992812],
};

function srgbToLinear(c: number): number {
  return c > 0.04045 ? Math.pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
}
function linearToSrgb(c: number): number {
  const v = c > 0.0031308 ? 1.055 * Math.pow(Math.max(c, 0), 1 / 2.4) - 0.055 : 12.92 * c;
  return clamp01(v);
}

/** Brettel dichromat simulation at full severity (comparison view). */
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

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}
