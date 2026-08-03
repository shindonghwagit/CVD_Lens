"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { useModel } from "../context/ModelContext";
import { CVDType } from "../hooks/useCVDModel";
import { simulate } from "@/lib/cvdSim";

const CVD_LABELS: Record<CVDType, string> = {
  p: "적색맹 (Protanopia)",
  d: "녹색맹 (Deuteranopia)",
  t: "청색맹 (Tritanopia)",
};

// Cap on the square side uploaded to the server. Kept equal to the backend's
// MAX_SIDE (cvd-lens/inference/main.py) — the server re-caps the long side to
// this anyway, so sending more only wastes bandwidth. Sync manually if changed.
const MAX_UPLOAD = 2048;
const THUMB_SIZE = 256;

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

function imageDataToURL(id: ImageData): string {
  const c = document.createElement("canvas");
  c.width = id.width; c.height = id.height;
  c.getContext("2d")!.putImageData(id, 0, 0);
  return c.toDataURL("image/jpeg", 0.92);
}

export default function ImageCorrection() {
  const { ready, error, infer } = useModel();
  const [cvdType, setCvdType] = useState<CVDType>("d");
  const [severity, setSeverity] = useState(1.0);
  const [original, setOriginal] = useState<string | null>(null);
  const [corrected, setCorrected] = useState<string | null>(null);
  const [showSim, setShowSim] = useState(false);
  const [simOrig, setSimOrig] = useState<string | null>(null);
  const [simOut, setSimOut] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved">("idle");
  const [sliderX, setSliderX] = useState(50);
  const [dragging, setDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Kept ImageData for re-inference on severity/type change (no file re-read)
  // and for client-side sim (no extra server round-trip).
  const sourceIDRef = useRef<ImageData | null>(null);
  const correctedIDRef = useRef<ImageData | null>(null);

  const computeSims = useCallback((type: CVDType) => {
    if (!sourceIDRef.current || !correctedIDRef.current) return;
    setSimOrig(imageDataToURL(simulate(sourceIDRef.current, type)));
    setSimOut(imageDataToURL(simulate(correctedIDRef.current, type)));
  }, []);

  const runInference = useCallback(async (type: CVDType, sev: number) => {
    if (!ready || !sourceIDRef.current) return;
    setProcessing(true);
    try {
      const result = await infer(sourceIDRef.current, type, sev);
      correctedIDRef.current = result;
      setCorrected(imageDataToURL(result));
      if (showSim) computeSims(type);
    } finally {
      setProcessing(false);
      setSaveState("idle");
    }
  }, [ready, infer, showSim, computeSims]);

  const processImage = useCallback(async (file: File, type: CVDType, sev: number) => {
    if (!ready) return;
    setProcessing(true);
    setCorrected(null);
    setSimOrig(null); setSimOut(null);

    const bitmap = await createImageBitmap(file);
    const side = Math.min(bitmap.width, bitmap.height);
    const target = Math.min(side, MAX_UPLOAD);   // cap the long side; never upscale a small image
    const canvas = document.createElement("canvas");
    canvas.width = target;
    canvas.height = target;
    const ctx = canvas.getContext("2d")!;

    const sx = (bitmap.width - side) / 2;
    const sy = (bitmap.height - side) / 2;
    ctx.drawImage(bitmap, sx, sy, side, side, 0, 0, target, target);

    setOriginal(canvas.toDataURL("image/jpeg", 0.95));   // JPEG (not PNG): a 2048² PNG dataURL is multi-MB
    sourceIDRef.current = ctx.getImageData(0, 0, target, target);
    await runInference(type, sev);
  }, [ready, runInference]);

  const saveCorrection = useCallback(async () => {
    if (!original || !corrected || saveState === "saving") return;
    setSaveState("saving");
    const [origThumb, corrThumb] = await Promise.all([
      resizeDataURL(original, THUMB_SIZE),
      resizeDataURL(corrected, THUMB_SIZE),
    ]);
    try {
      const res = await fetch("/api/corrections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cvdType, originalImage: origThumb, correctedImage: corrThumb }),
      });
      setSaveState(res.ok ? "saved" : "idle");
    } catch {
      setSaveState("idle");
    }
  }, [original, corrected, cvdType, saveState]);

  const onFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    processImage(file, cvdType, severity);
  }, [processImage, cvdType, severity]);

  const pendingFileRef = useRef<File | null>(null);
  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) { pendingFileRef.current = file; onFile(file); }
  };

  // Re-infer on CVD type change (reuse stored source; no file re-read).
  const prevCvdType = useRef(cvdType);
  useEffect(() => {
    if (prevCvdType.current === cvdType) return;
    prevCvdType.current = cvdType;
    if (sourceIDRef.current && ready) runInference(cvdType, severity);
  }, [cvdType, ready, severity, runInference]);

  // Debounced re-infer on severity change (~300ms — server round-trip).
  const sevTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onSeverity = useCallback((v: number) => {
    setSeverity(v);
    if (!sourceIDRef.current) return;
    if (sevTimer.current) clearTimeout(sevTimer.current);
    sevTimer.current = setTimeout(() => runInference(cvdType, v), 300);
  }, [cvdType, runInference]);

  // Recompute sim views when toggled on (or when corrected changes while on).
  useEffect(() => {
    if (showSim) computeSims(cvdType);
    else { setSimOrig(null); setSimOut(null); }
  }, [showSim, corrected, cvdType, computeSims]);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) { pendingFileRef.current = file; onFile(file); }
  };

  const onSliderMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    setSliderX(x);
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent | TouchEvent) => {
      const x = "touches" in e ? e.touches[0].clientX : e.clientX;
      onSliderMove(x);
    };
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

  return (
    <div className="flex flex-col items-center gap-6">
      {/* CVD 타입 선택 */}
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

      {/* 업로드 영역 */}
      {!original && (
        <div
          onDrop={onDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => fileInputRef.current?.click()}
          className="w-full max-w-sm h-48 border-2 border-dashed rounded-xl flex flex-col items-center justify-center gap-2 cursor-pointer transition-colors"
          style={{ borderColor: "var(--border-strong)", color: "var(--fg-subtle)" }}
          onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--color-brand)")}
          onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border-strong)")}
        >
          <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <p className="text-sm" style={{ color: "var(--fg-muted)" }}>이미지를 드래그하거나 클릭해서 업로드</p>
          <p className="text-xs" style={{ color: "var(--fg-subtle)" }}>JPG, PNG 지원</p>
          <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={onFileChange} />
        </div>
      )}

      {/* 비교 슬라이더 + 컨트롤 */}
      {original && (
        <div className="flex flex-col items-center gap-4 w-full max-w-sm">
          <div
            ref={containerRef}
            className="relative w-full max-w-lg aspect-square rounded-xl overflow-hidden cursor-ew-resize select-none border"
            style={{ borderColor: "var(--border)" }}
            onMouseDown={() => setDragging(true)}
            onTouchStart={() => setDragging(true)}
          >
            {corrected && (
              <img src={corrected} alt="corrected" className="absolute inset-0 w-full h-full object-cover" draggable={false} />
            )}
            {original && (
              <div className="absolute inset-0 overflow-hidden" style={{ width: `${sliderX}%` }}>
                <img src={original} alt="original" className="absolute inset-0 w-full h-full max-w-none object-cover" draggable={false} />
              </div>
            )}
            <div className="absolute top-0 bottom-0 w-0.5 shadow-lg" style={{ left: `${sliderX}%`, background: "var(--bg-elevated)" }}>
              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full shadow-lg flex items-center justify-center" style={{ background: "var(--bg-elevated)" }}>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ color: "var(--fg)" }}>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l-4 3 4 3M16 9l4 3-4 3" />
                </svg>
              </div>
            </div>
            <span className="absolute top-2 left-2 text-xs bg-black/50 text-white px-2 py-0.5 rounded-full">원본</span>
            <span className="absolute top-2 right-2 text-xs px-2 py-0.5 rounded-full text-white" style={{ background: "var(--color-brand)" }}>보정</span>

            {processing && (
              <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                <p className="text-white text-sm">처리 중...</p>
              </div>
            )}
          </div>

          {/* severity 슬라이더 */}
          <div className="w-full flex items-center gap-3">
            <span className="text-xs whitespace-nowrap" style={{ color: "var(--fg-muted)" }}>보정 강도</span>
            <input
              type="range" min={0} max={1} step={0.05} value={severity}
              onChange={(e) => onSeverity(parseFloat(e.target.value))}
              className="flex-1 accent-[var(--color-brand)]"
              aria-label="보정 강도"
            />
            <span className="text-xs font-mono w-9 text-right" style={{ color: "var(--fg)" }}>
              {severity.toFixed(2)}
            </span>
          </div>

          {/* CVD 시뮬레이션 보기 토글 */}
          <label className="flex items-center gap-2 text-sm cursor-pointer" style={{ color: "var(--fg-muted)" }}>
            <input type="checkbox" checked={showSim} onChange={(e) => setShowSim(e.target.checked)} className="accent-[var(--color-brand)]" />
            CVD 시뮬레이션 보기
          </label>

          {/* sim 비교 (같은 화면 나란히) */}
          {showSim && (
            <div className="w-full grid grid-cols-2 gap-3">
              <figure className="flex flex-col gap-1">
                <div className="aspect-square rounded-lg overflow-hidden border" style={{ borderColor: "var(--border)" }}>
                  {simOrig && <img src={simOrig} alt="sim original" className="w-full h-full object-cover" />}
                </div>
                <figcaption className="text-[11px] text-center" style={{ color: "var(--fg-subtle)" }}>
                  색각이상자가 보는 원본
                </figcaption>
              </figure>
              <figure className="flex flex-col gap-1">
                <div className="aspect-square rounded-lg overflow-hidden border" style={{ borderColor: "var(--border)" }}>
                  {simOut && <img src={simOut} alt="sim corrected" className="w-full h-full object-cover" />}
                </div>
                <figcaption className="text-[11px] text-center" style={{ color: "var(--fg-subtle)" }}>
                  색각이상자가 보는 보정본
                </figcaption>
              </figure>
            </div>
          )}

          <div className="flex items-center gap-3">
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
              onClick={() => { setOriginal(null); setCorrected(null); setSaveState("idle"); setShowSim(false); sourceIDRef.current = null; correctedIDRef.current = null; pendingFileRef.current = null; fileInputRef.current && (fileInputRef.current.value = ""); }}
              className="text-sm transition-colors"
              style={{ color: "var(--fg-subtle)" }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "var(--fg)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "var(--fg-subtle)")}
            >
              다른 이미지 선택
            </button>
          </div>
        </div>
      )}

      {!ready && <p className="text-sm" style={{ color: "#d89a2b" }}>서버 연결 중...</p>}
      {error && <p className="text-sm" style={{ color: "#d5383a" }}>오류: {error}</p>}
    </div>
  );
}
