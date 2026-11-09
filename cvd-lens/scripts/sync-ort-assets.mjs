// Keep the public WASM loader/binary from the same build as the JS dependency.
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";

const require = createRequire(import.meta.url);
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dist = dirname(require.resolve("onnxruntime-web"));
const { version } = JSON.parse(readFileSync(resolve(dist, "../package.json"), "utf8"));
const target = resolve(root, "public/ort");
mkdirSync(target, { recursive: true });
let copied = 0;
for (const extension of ["mjs", "wasm"]) {
  const name = `ort-wasm-simd-threaded.${extension}`;
  const source = readFileSync(resolve(dist, name));
  const destination = resolve(target, name);
  if (!existsSync(destination) || !source.equals(readFileSync(destination))) {
    writeFileSync(destination, source);
    copied++;
  }
}
console.log(`ONNX Runtime Web ${version}: ${copied} WASM assets synchronized`);
