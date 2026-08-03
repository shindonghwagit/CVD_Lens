"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CVDType, useCVDModel } from "../hooks/useCVDModel";
import { simulate } from "@/lib/cvdSim";

function imageDataToURL(id: ImageData): string {
  const c = document.createElement("canvas");
  c.width = id.width; c.height = id.height;
  c.getContext("2d")!.putImageData(id, 0, 0);
  return c.toDataURL("image/jpeg", 0.92);
}

// Reject if the inference request hasn't resolved within `ms`. Client-side
// guard for a surfaced error state; the request contract itself is unchanged.
function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    p,
    new Promise<T>((_, reject) => setTimeout(() => reject(new Error("timeout")), ms)),
  ]);
}
const REQUEST_TIMEOUT = 30000;

// Cap on the captured square side. Kept equal to the backend's MAX_SIDE
// (cvd-lens/inference/main.py). Sync manually if changed.
const MAX_UPLOAD = 2048;

const CVD_LABELS: Record<CVDType, string> = {
  p: "적색맹 (Protanopia)",
  d: "녹색맹 (Deuteranopia)",
  t: "청색맹 (Tritanopia)",
};

function resizeDataURL(src: string, size: number): Promise<string> {
  return new Promise((resolve) => {
    const img = document.createElement("img");
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = size; c.height = size;
      c.getContext("2d")!.drawImage(img, 0, 0, size, size);
      resolve(c.toDataURL("image/jpeg", 0.75));
    };
    img.src = src;
  });
}

export default function CameraView() {
  const videoRef    = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [cvdType, setCvdType]     = useState<CVDType>("d");
  const [streaming, setStreaming] = useState(false);
  const [original, setOriginal]   = useState<string | null>(null);
  const [corrected, setCorrected] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [slowLoad, setSlowLoad]   = useState(false);
  const [reqError, setReqError]   = useState(false);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved">("idle");
  const [compareMode, setCompareMode] = useState<"side" | "overlay">("side");
  const [sliderX, setSliderX]     = useState(50);
  const [dragging, setDragging]   = useState(false);
  const [showSim, setShowSim]     = useState(false);
  const [simOrig, setSimOrig]     = useState<string | null>(null);
  const [simOut, setSimOut]       = useState<string | null>(null);

  const { ready, error, infer } = useCVDModel();

  // Kept ImageData for client-side sim (no extra server round-trip).
  const sourceIDRef = useRef<ImageData | null>(null);
  const correctedIDRef = useRef<ImageData | null>(null);

  const computeSims = useCallback((type: CVDType) => {
    if (!sourceIDRef.current || !correctedIDRef.current) return;
    setSimOrig(imageDataToURL(simulate(sourceIDRef.current, type)));
    setSimOut(imageDataToURL(simulate(correctedIDRef.current, type)));
  }, []);

  useEffect(() => {
    if (showSim) computeSims(cvdType);
    else { setSimOrig(null); setSimOut(null); }
  }, [showSim, corrected, cvdType, computeSims]);

  // Cold-start hint: after 5s of processing, swap the overlay copy so the user
  // knows a sleeping Render instance may take up to ~1min on the first request.
  useEffect(() => {
    if (!processing) { setSlowLoad(false); return; }
    const t = setTimeout(() => setSlowLoad(true), 5000);
    return () => clearTimeout(t);
  }, [processing]);

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      videoRef.current!.srcObject = stream;
      await videoRef.current!.play();
      setStreaming(true);
    } catch {
      alert("카메라 접근에 실패했습니다.");
    }
  }

  function stopCamera() {
    const video = videoRef.current;
    if (video?.srcObject) {
      (video.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    setStreaming(false);
  }

  const capture = useCallback(async () => {
    const video = videoRef.current;
    if (!video || !ready) return;
    setReqError(false);
    setProcessing(true);
    setCorrected(null);

    const vw = video.videoWidth, vh = video.videoHeight;
    const side = Math.min(vw, vh);
    const target = Math.min(side, MAX_UPLOAD);   // cap; camera frames are usually well under this
    const canvas = document.createElement("canvas");
    canvas.width = target; canvas.height = target;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(video, (vw - side) / 2, (vh - side) / 2, side, side, 0, 0, target, target);

    setOriginal(canvas.toDataURL("image/jpeg", 0.92));
    stopCamera();

    try {
      const imageData = ctx.getImageData(0, 0, target, target);
      sourceIDRef.current = imageData;
      const result = await withTimeout(infer(imageData, cvdType), REQUEST_TIMEOUT);
      correctedIDRef.current = result;
      ctx.putImageData(result, 0, 0);
      setCorrected(canvas.toDataURL("image/jpeg", 0.92));
      if (showSim) computeSims(cvdType);
      setSaveState("idle");
    } catch (e) {
      console.error("보정 오류:", e);
      setReqError(true);
    }
    setProcessing(false);
  }, [ready, cvdType, infer, showSim, computeSims]);

  // Re-run inference on the last captured frame (no re-capture needed).
  const retry = useCallback(async () => {
    if (!sourceIDRef.current || !ready) return;
    setReqError(false);
    setProcessing(true);
    try {
      const result = await withTimeout(infer(sourceIDRef.current, cvdType), REQUEST_TIMEOUT);
      correctedIDRef.current = result;
      setCorrected(imageDataToURL(result));
      if (showSim) computeSims(cvdType);
      setSaveState("idle");
    } catch (e) {
      console.error("보정 요청 실패:", e);
      setReqError(true);
    }
    setProcessing(false);
  }, [ready, cvdType, infer, showSim, computeSims]);

  // Download the corrected JPEG. Camera captures have no source filename,
  // so the base is a fixed "capture" — capture_corrected_{type}.jpg.
  const downloadCorrected = useCallback(() => {
    if (!corrected) return;
    const a = document.createElement("a");
    a.href = corrected;
    a.download = `capture_corrected_${cvdType}.jpg`;
    a.click();
  }, [corrected, cvdType]);

  const saveCorrection = useCallback(async () => {
    if (!original || !corrected || saveState === "saving") return;
    setSaveState("saving");
    const [origThumb, corrThumb] = await Promise.all([
      resizeDataURL(original, 256),
      resizeDataURL(corrected, 256),
    ]);
    try {
      const res = await fetch("/api/corrections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cvdType, source: "camera", originalImage: origThumb, correctedImage: corrThumb }),
      });
      setSaveState(res.ok ? "saved" : "idle");
    } catch {
      setSaveState("idle");
    }
  }, [original, corrected, cvdType, saveState]);

  const onSliderMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    setSliderX(Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100)));
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent | TouchEvent) =>
      onSliderMove("touches" in e ? e.touches[0].clientX : e.clientX);
    const onUp = () => setDragging(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchmove", onMove);
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onUp);
    };
  }, [dragging, onSliderMove]);

  const processingOverlay = (
    <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center gap-3 px-4 text-center">
      <div className="w-6 h-6 rounded-full border-2 border-white/30 border-t-white animate-spin" />
      <p className="text-white text-sm leading-snug">
        {slowLoad ? "서버를 깨우는 중입니다 — 첫 요청은 최대 1분 걸릴 수 있어요" : "처리 중..."}
      </p>
    </div>
  );

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="flex gap-2 flex-wrap justify-center">
        {(Object.keys(CVD_LABELS) as CVDType[]).map((type) => (
          <button
            key={type}
            onClick={() => setCvdType(type)}
            className="px-4 py-2 rounded-full text-sm font-medium transition-colors"
            style={{
              background: cvdType === type ? "var(--color-brand)" : "var(--bg-muted)",
              color: cvdType === type ? "#ffffff" : "var(--fg-muted)",
              border: "1px solid",
              borderColor: cvdType === type ? "var(--color-brand)" : "var(--border)",
            }}
          >
            {CVD_LABELS[type]}
          </button>
        ))}
      </div>

      {!original ? (
        <div className="flex flex-col items-center gap-4 w-full">
          <div className="relative w-full max-w-sm aspect-square rounded-xl overflow-hidden border" style={{ background: "#000", borderColor: "var(--border)" }}>
            <video ref={videoRef} className="w-full h-full object-cover" playsInline muted />
            {!streaming && (
              <div className="absolute inset-0 flex items-center justify-center">
                <p className="text-sm" style={{ color: "var(--fg-subtle)" }}>카메라가 꺼져 있습니다</p>
              </div>
            )}
          </div>
          <div className="flex gap-3">
            {!streaming ? (
              <button
                onClick={startCamera}
                disabled={!ready}
                className="px-6 py-2.5 rounded-full text-white text-sm font-medium transition-colors disabled:opacity-40"
                style={{ background: "var(--color-brand)" }}
              >
                카메라 시작
              </button>
            ) : (
              <>
                <button
                  onClick={capture}
                  className="px-6 py-2.5 rounded-full text-sm font-semibold transition-colors"
                  style={{ background: "var(--fg)", color: "var(--bg)" }}
                >
                  촬영
                </button>
                <button
                  onClick={stopCamera}
                  className="px-6 py-2.5 rounded-full text-sm font-medium transition-colors"
                  style={{ background: "var(--bg-muted)", color: "var(--fg-muted)", border: "1px solid var(--border)" }}
                >
                  취소
                </button>
              </>
            )}
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-4 w-full max-w-2xl">
          {/* 비교 모드 토글: 나란히(기본) ↔ 겹쳐 보기(wipe) */}
          <div className="inline-flex p-0.5 rounded-full border" style={{ background: "var(--bg-muted)", borderColor: "var(--border)" }}>
            {(["side", "overlay"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setCompareMode(mode)}
                className="px-4 py-1.5 rounded-full text-xs font-medium transition-colors"
                style={{
                  background: compareMode === mode ? "var(--bg-elevated)" : "transparent",
                  color: compareMode === mode ? "var(--fg)" : "var(--fg-muted)",
                  boxShadow: compareMode === mode ? "var(--shadow-soft)" : "none",
                }}
              >
                {mode === "side" ? "나란히" : "겹쳐 보기"}
              </button>
            ))}
          </div>

          {compareMode === "side" ? (
            /* 나란히: 왼쪽=원본, 오른쪽=보정 (모바일 폭에서는 세로 스택) */
            <div className="w-full grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="relative aspect-square rounded-xl overflow-hidden border" style={{ borderColor: "var(--border)" }}>
                {original && <img src={original} alt="original" className="absolute inset-0 w-full h-full object-cover" draggable={false} />}
                <span className="absolute top-2 left-2 text-xs bg-black/50 text-white px-2 py-0.5 rounded-full">원본</span>
              </div>
              <div className="relative aspect-square rounded-xl overflow-hidden border" style={{ borderColor: "var(--border)" }}>
                {corrected && <img src={corrected} alt="corrected" className="absolute inset-0 w-full h-full object-cover" draggable={false} />}
                <span className="absolute top-2 right-2 text-xs px-2 py-0.5 rounded-full text-white" style={{ background: "var(--color-brand)" }}>보정</span>
                {processing && processingOverlay}
              </div>
            </div>
          ) : (
            /* 겹쳐 보기: wipe 슬라이더 (세밀 비교용) */
            <div
              ref={containerRef}
              className="relative w-full max-w-lg aspect-square rounded-xl overflow-hidden cursor-ew-resize select-none border"
              style={{ borderColor: "var(--border)" }}
              onMouseDown={() => setDragging(true)}
              onTouchStart={() => setDragging(true)}
            >
              {corrected && <img src={corrected} alt="corrected" className="absolute inset-0 w-full h-full object-cover" draggable={false} />}
              <div className="absolute inset-0 overflow-hidden" style={{ width: `${sliderX}%` }}>
                <img src={original!} alt="original" className="absolute inset-0 w-full h-full max-w-none object-cover" draggable={false} />
              </div>
              <div className="absolute top-0 bottom-0 w-0.5 shadow-lg" style={{ left: `${sliderX}%`, background: "var(--bg-elevated)" }}>
                <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full shadow-lg flex items-center justify-center" style={{ background: "var(--bg-elevated)" }}>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ color: "var(--fg)" }}>
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l-4 3 4 3M16 9l4 3-4 3" />
                  </svg>
                </div>
              </div>
              <span className="absolute top-2 left-2 text-xs bg-black/50 text-white px-2 py-0.5 rounded-full">원본</span>
              <span className="absolute top-2 right-2 text-xs px-2 py-0.5 rounded-full text-white" style={{ background: "var(--color-brand)" }}>보정</span>
              {processing && processingOverlay}
            </div>
          )}

          {/* CVD 시뮬레이션 보기 토글 */}
          <label className="flex items-center gap-2 text-sm cursor-pointer" style={{ color: "var(--fg-muted)" }}>
            <input type="checkbox" checked={showSim} onChange={(e) => setShowSim(e.target.checked)} className="accent-[var(--color-brand)]" />
            CVD 시뮬레이션 보기
          </label>
          {showSim && (
            <div className="w-full grid grid-cols-2 gap-3">
              <figure className="flex flex-col gap-1">
                <div className="aspect-square rounded-lg overflow-hidden border" style={{ borderColor: "var(--border)" }}>
                  {simOrig && <img src={simOrig} alt="sim original" className="w-full h-full object-cover" />}
                </div>
                <figcaption className="text-[11px] text-center" style={{ color: "var(--fg-subtle)" }}>색각이상자가 보는 원본</figcaption>
              </figure>
              <figure className="flex flex-col gap-1">
                <div className="aspect-square rounded-lg overflow-hidden border" style={{ borderColor: "var(--border)" }}>
                  {simOut && <img src={simOut} alt="sim corrected" className="w-full h-full object-cover" />}
                </div>
                <figcaption className="text-[11px] text-center" style={{ color: "var(--fg-subtle)" }}>색각이상자가 보는 보정본</figcaption>
              </figure>
            </div>
          )}

          {reqError && !processing && (
            <div
              className="w-full flex flex-col items-center gap-2 rounded-lg border px-4 py-3 text-center"
              style={{ background: "#d5383a10", borderColor: "#d5383a44" }}
            >
              <p className="text-sm" style={{ color: "#d5383a" }}>서버 연결에 실패했어요. 잠시 후 다시 시도해주세요</p>
              <button
                onClick={retry}
                className="px-4 py-1.5 rounded-full text-sm font-medium text-white transition-colors"
                style={{ background: "var(--color-brand)" }}
              >
                다시 시도
              </button>
            </div>
          )}

          <div className="flex items-center gap-3 flex-wrap justify-center">
            {corrected && !processing && (
              <button
                onClick={downloadCorrected}
                className="px-4 py-2 rounded-full text-sm font-medium transition-colors text-white"
                style={{ background: "var(--color-brand)" }}
              >
                이미지 저장
              </button>
            )}
            {corrected && !processing && (
              <button
                onClick={saveCorrection}
                disabled={saveState === "saving" || saveState === "saved"}
                className="px-4 py-2 rounded-full text-sm font-medium transition-colors disabled:opacity-60"
                style={saveState === "saved"
                  ? { background: "#22c55e18", color: "#22c55e", border: "1px solid #22c55e44" }
                  : { background: "var(--bg-muted)", color: "var(--fg)", border: "1px solid var(--border-strong)" }
                }
              >
                {saveState === "saving" ? "저장 중..." : saveState === "saved" ? "저장됨 ✓" : "목록에 저장"}
              </button>
            )}
            <button
              onClick={() => { setOriginal(null); setCorrected(null); setSaveState("idle"); setShowSim(false); setReqError(false); sourceIDRef.current = null; correctedIDRef.current = null; }}
              className="text-sm transition-colors"
              style={{ color: "var(--fg-subtle)" }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "var(--fg)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "var(--fg-subtle)")}
            >
              다시 촬영
            </button>
          </div>
        </div>
      )}

      {!ready && <p className="text-sm" style={{ color: "#d89a2b" }}>서버 연결 중...</p>}
      {error && <p className="text-sm" style={{ color: "#d5383a" }}>오류: {error}</p>}
    </div>
  );
}
