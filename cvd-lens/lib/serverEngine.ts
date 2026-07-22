/**
 * serverEngine — PRESERVED alternative inference path (NOT the default).
 *
 * This was the original `useCVDModel` implementation: it POSTed each frame to a
 * FastAPI server (`inference/main.py`) that ran a server-side ONNX model
 * (cvdlens_fp32.onnx, the pre-pivot 4-channel model). Phase 2 replaced this with
 * in-browser inference (see lib/cvdEngine.ts).
 *
 * Kept intact because server-side ONNX on a GPU box remains a valid stage-1
 * deployment option (e.g. if a future model is too heavy for the browser). To
 * re-enable, wire `pingServer` / `inferViaServer` back into useCVDModel.
 *
 * Requires NEXT_PUBLIC_API_URL (default http://localhost:8000) and the FastAPI
 * service running.
 */
import type { CVDType } from "./cvdEngine";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Health-check the inference server (used as the old `ready` signal). */
export async function pingServer(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

/** Original server round-trip: ImageData → JPEG → POST /infer → JPEG → ImageData. */
export async function inferViaServer(
  imageData: ImageData,
  cvdType: CVDType
): Promise<ImageData> {
  const canvas = document.createElement("canvas");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d")!.putImageData(imageData, 0, 0);

  const blob = await new Promise<Blob>((resolve) =>
    canvas.toBlob((b) => resolve(b!), "image/jpeg", 0.85)
  );

  const form = new FormData();
  form.append("image", blob, "frame.jpg");
  form.append("cvd_type", cvdType);

  const res = await fetch(`${API_URL}/infer`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`서버 오류: ${res.status}`);

  const resBlob = await res.blob();
  const bitmap = await createImageBitmap(resBlob);
  const outCanvas = document.createElement("canvas");
  outCanvas.width = imageData.width;
  outCanvas.height = imageData.height;
  outCanvas.getContext("2d")!.drawImage(bitmap, 0, 0, imageData.width, imageData.height);
  return outCanvas.getContext("2d")!.getImageData(0, 0, imageData.width, imageData.height);
}
