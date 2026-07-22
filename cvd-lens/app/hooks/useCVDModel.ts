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
    severity = 1.0
  ): Promise<ImageData> => {
    // ImageData → JPEG Blob
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
    form.append("severity", String(severity));

    const res = await fetch(`${API_URL}/infer`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`서버 오류: ${res.status}`);

    // JPEG 응답 → ImageData
    const resBlob = await res.blob();
    const bitmap = await createImageBitmap(resBlob);
    const outCanvas = document.createElement("canvas");
    outCanvas.width = imageData.width;
    outCanvas.height = imageData.height;
    outCanvas.getContext("2d")!.drawImage(bitmap, 0, 0, imageData.width, imageData.height);
    return outCanvas.getContext("2d")!.getImageData(0, 0, imageData.width, imageData.height);
  }, []);

  return { ready, error, infer };
}
