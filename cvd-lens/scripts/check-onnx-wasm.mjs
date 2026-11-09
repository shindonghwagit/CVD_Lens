// Actual browser WASM bundle + public runtime assets + both shipped model files.
// Node supports this single-threaded WASM backend; this is not a browser UI test.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

const require = createRequire(import.meta.url);
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dist = dirname(require.resolve("onnxruntime-web"));
assert.match(readFileSync(resolve(root, "lib/cvdOnnx.ts"), "utf8"), /import\(["']onnxruntime-web\/wasm["']\)/,
  "Video inference must use the WASM-only browser entry");
for (const extension of ["mjs", "wasm"]) {
  const name = `ort-wasm-simd-threaded.${extension}`;
  assert.ok(readFileSync(resolve(root, "public/ort", name)).equals(readFileSync(resolve(dist, name))),
    `Stale public runtime: ${name}; run npm run prebuild`);
}
const runtimePackage = JSON.parse(readFileSync(resolve(dist, "../package.json"), "utf8"));
const browserEntry = runtimePackage.exports["./wasm"].import.default;
const ort = await import(pathToFileURL(resolve(dist, "..", browserEntry)).href);
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = {
  mjs: pathToFileURL(resolve(root, "public/ort/ort-wasm-simd-threaded.mjs")).href,
  wasm: pathToFileURL(resolve(root, "public/ort/ort-wasm-simd-threaded.wasm")).href,
};
const size = 256, pixels = size * size;
const input = new Float32Array(3 * pixels);
const rgba = new Uint8ClampedArray(4 * pixels);
for (let y = 0; y < size; y++) {
  for (let x = 0; x < size; x++) {
    const i = y * size + x;
    input[i] = x / 255;
    input[pixels + i] = y / 255;
    input[2 * pixels + i] = ((x + y) % 256) / 255;
    rgba.set([x, y, (x + y) % 256, 255], 4 * i);
  }
}
const expected = new Map();
for (const type of ["p", "d"]) {
  const session = await ort.InferenceSession.create(
    new Uint8Array(readFileSync(resolve(root, `public/models/cvdlens_${type}.onnx`))),
    { executionProviders: ["wasm"], graphOptimizationLevel: "all" },
  );
  const outputs = await session.run({
    srgb: new ort.Tensor("float32", input, [1, 3, size, size]),
    severity: new ort.Tensor("float32", new Float32Array([1]), [1, 1]),
  });
  assert.deepEqual(outputs.out_srgb.dims, [1, 3, size, size]);
  assert.ok(["float32", "float64"].includes(outputs.out_srgb.type));
  const output = Float32Array.from(outputs.out_srgb.data);
  assert.equal(output.length, input.length);
  let delta = 0;
  for (let i = 0; i < output.length; i++) {
    assert.ok(Number.isFinite(output[i]) && output[i] >= 0 && output[i] <= 1);
    delta += Math.abs(output[i] - input[i]);
  }
  delta /= output.length;
  assert.ok(delta > 1e-5, `${type}: correction unexpectedly identity`);
  console.log(`PASS ${type}: browser WASM ${ort.env.versions.web}, session + inference, MAE=${delta.toFixed(6)}`);
  expected.set(type, output);
  await session.release();
}

// Exercise the actual app's preload/cache/RGBA conversion/output-normalization code.
const ts = require("typescript");
const compiled = ts.transpileModule(readFileSync(resolve(root, "lib/cvdOnnx.ts"), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const modelRequests = [];
const appRuntime = { ...ort, InferenceSession: {
  create: (url, options) => {
    // Node has no browser origin. Resolve only the app's two static model URLs
    // to the same public bytes; session creation/inference still use real WASM.
    assert.match(url, /^\/models\/cvdlens_[pd]\.onnx$/);
    modelRequests.push(url);
    return ort.InferenceSession.create(new Uint8Array(readFileSync(resolve(root, `public${url}`))), options);
  },
} };
const context = { exports: {}, Float32Array, Float64Array, Uint8ClampedArray, URL,
  window: { location: { origin: "https://cvdlens.test" } },
  require: (name) => {
    assert.equal(name, "onnxruntime-web/wasm");
    return appRuntime;
  } };
const localWasmPaths = ort.env.wasm.wasmPaths;
runInNewContext(compiled, context);
assert.equal(modelRequests.length, 0, "Server rendering must not load a model or initialize WASM");
await context.exports.preloadSession("p");
assert.equal(ort.env.wasm.wasmPaths.mjs,
  `https://cvdlens.test/ort/ort-wasm-simd-threaded.mjs?v=${ort.env.versions.web}`);
assert.equal(ort.env.wasm.wasmPaths.wasm,
  `https://cvdlens.test/ort/ort-wasm-simd-threaded.wasm?v=${ort.env.versions.web}`);
ort.env.wasm.wasmPaths = localWasmPaths;
for (const type of ["p", "d"]) {
  await context.exports.preloadSession(type);
  const output = await context.exports.runOnnxCorrection(type, { data: rgba, width: size, height: size }, 1);
  assert.ok(output instanceof Float32Array);
  assert.deepEqual(output, expected.get(type));
  console.log(`PASS ${type}: actual video inference wrapper matches the browser WASM output`);
}
assert.deepEqual(modelRequests, ["/models/cvdlens_p.onnx", "/models/cvdlens_d.onnx"],
  "Preload and inference must share one cached session per type");
