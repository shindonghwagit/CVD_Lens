"use client";

import { useCallback, useEffect, useState } from "react";

export type CVDType = "p" | "d" | "t";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function useCVDModel() {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 서버 상태 확인
  useEffect(() => {
    let cancelled = false;
    let retryTimeout: ReturnType<typeof setTimeout>;

    async function ping() {
      try {
        const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(3000) });
        if (!cancelled && res.ok) {
          setReady(true);
          setError(null);
          return;
        }
      } catch {
        // 재시도
      }
      if (!cancelled) {
        setError("서버에 연결할 수 없습니다. (localhost:8000)");
        retryTimeout = setTimeout(ping, 3000);
      }
    }

    ping();
    return () => {
      cancelled = true;
      clearTimeout(retryTimeout);
    };
  }, []);

  const infer = useCallback(async (
    imageData: ImageData,
    cvdType: CVDType,
    strength = 1.0,
    signal?: AbortSignal,
  ): Promise<ImageData> => {
    // ImageData → JPEG Blob
    const canvas = document.createElement("canvas");
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    canvas.getContext("2d")!.putImageData(imageData, 0, 0);

    const blob = await new Promise<Blob>((resolve) =>
      canvas.toBlob((b) => resolve(b!), "image/jpeg", 0.92)   // match server JPEG q92; minimize upload-side loss
    );

    const form = new FormData();
    form.append("image", blob, "frame.jpg");
    form.append("cvd_type", cvdType);
    // P/D models were trained at severity=1.0.  Keep inference on the
    // distribution seen during training and expose a separate display
    // strength by blending the learned correction delta below.
    form.append("severity", String(cvdType === "t" ? strength : 1.0));

    const res = await fetch(`${API_URL}/infer`, { method: "POST", body: form, signal });
    if (!res.ok) throw new Error(`서버 오류: ${res.status}`);

    // JPEG 응답 → ImageData
    const resBlob = await res.blob();
    const bitmap = await createImageBitmap(resBlob);
    const outCanvas = document.createElement("canvas");
    outCanvas.width = imageData.width;
    outCanvas.height = imageData.height;
    const outCtx = outCanvas.getContext("2d")!;
    outCtx.drawImage(bitmap, 0, 0, imageData.width, imageData.height);
    const output = outCtx.getImageData(0, 0, imageData.width, imageData.height);
    bitmap.close();

    if (cvdType !== "t" && strength < 1) {
      const amount = Math.max(0, strength);
      for (let i = 0; i < output.data.length; i += 4) {
        output.data[i] = imageData.data[i] + amount * (output.data[i] - imageData.data[i]);
        output.data[i + 1] = imageData.data[i + 1] + amount * (output.data[i + 1] - imageData.data[i + 1]);
        output.data[i + 2] = imageData.data[i + 2] + amount * (output.data[i + 2] - imageData.data[i + 2]);
      }
    }
    return output;
  }, []);

  return { ready, error, infer };
}
