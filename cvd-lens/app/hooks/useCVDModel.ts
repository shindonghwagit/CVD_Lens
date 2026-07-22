"use client";

import { useCallback, useEffect, useState } from "react";
import type { CVDType } from "@/lib/cvdEngine";

export type { CVDType };

/**
 * useCVDModel — IN-BROWSER inference (Phase 2 default).
 *
 * Was a FastAPI server round-trip; that path is preserved in lib/serverEngine.ts
 * as a documented fallback ("browser inference default, server inference
 * alternative"). cvdEngine (ort-web) is dynamically imported so it is NOT in the
 * landing bundle and only loads when a correction surface actually runs.
 *
 * `infer` gained a `severity` arg (default 1.0) — web UI slider value.
 */
export function useCVDModel() {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backend, setBackend] = useState<string>("");
  const [lastMs, setLastMs] = useState<number | null>(null);

  // Browser inference is available as soon as we're on the client. No model is
  // loaded here (would pull ~2.5 MB on the landing page); sessions load lazily.
  useEffect(() => {
    setReady(true);
  }, []);

  /** Optional warm-up: load a type's session ahead of first use (call on a
   *  correction surface's mount, never on the landing). */
  const preload = useCallback(async (type: CVDType) => {
    try {
      const eng = await import("@/lib/cvdEngine");
      setBackend(await eng.init(type));
      setError(null);
    } catch (e) {
      setError(`모델 로드 실패: ${(e as Error)?.message ?? String(e)}`);
    }
  }, []);

  const infer = useCallback(
    async (imageData: ImageData, cvdType: CVDType, severity = 1.0): Promise<ImageData> => {
      try {
        const eng = await import("@/lib/cvdEngine");
        const { image, ms, backend: b } = await eng.correct(imageData, cvdType, severity);
        setBackend(b);
        setLastMs(ms);
        setError(null);
        return image;
      } catch (e) {
        setError(`추론 실패: ${(e as Error)?.message ?? String(e)}`);
        throw e;
      }
    },
    []
  );

  /** Brettel CVD simulation (comparison view). */
  const simulate = useCallback(
    async (imageData: ImageData, cvdType: CVDType): Promise<ImageData> => {
      const eng = await import("@/lib/cvdEngine");
      return eng.simulate(imageData, cvdType);
    },
    []
  );

  return { ready, error, backend, lastMs, preload, infer, simulate };
}
