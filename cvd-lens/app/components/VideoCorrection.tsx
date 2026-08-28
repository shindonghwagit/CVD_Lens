"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CVDType } from "../hooks/useCVDModel";

const CVD_LABELS: Record<CVDType, string> = {
  p: "적색맹 (Protanopia)",
  d: "녹색맹 (Deuteranopia)",
  t: "청색맹 (Tritanopia)",
};

const MAX_DIM = 640;          // 실시간 캔버스 최대 변 (성능)
const TRITAN_DEG = 30;        // OpenCV H 단위(0–179), severity 1.0 기준. 배포 서버 _tritan_hue_shift와 동일.

/** 청색맹(t) 채도보존 hue 회전 — 배포 서버 _tritan_hue_shift를 픽셀 단위로 이식(OpenCV HSV 규약).
 *  blue(H 90–135)→violet / yellow(H 18–40)→yellow-green, S·V 고정. 인라인 RGB↔HSV로 실시간 처리. */
function correctTritan(data: Uint8ClampedArray) {
  const deg = TRITAN_DEG;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const v = Math.max(r, g, b), mn = Math.min(r, g, b), diff = v - mn;
    if (diff === 0) continue;                                   // 무채색 → 게이트 0, 건너뜀
    const s = (diff * 255) / v;                                 // OpenCV S (0–255)
    // hue (OpenCV 0–179)
    let hue: number;
    if (v === r) hue = 60 * (g - b) / diff;
    else if (v === g) hue = 120 + 60 * (b - r) / diff;
    else hue = 240 + 60 * (r - g) / diff;
    hue = hue / 2; if (hue < 0) hue += 180;
    // gate: 채도 floor + 청/황 밴드
    const satG = Math.min(Math.max((s - 18) / 50, 0), 1);
    if (satG === 0) continue;
    const band = (x: number, lo: number, hi: number) =>
      Math.min(Math.max((x - lo) / 12, 0), 1) * Math.min(Math.max((hi - x) / 12, 0), 1);
    const gate = satG * Math.max(band(hue, 90, 135), band(hue, 18, 40));
    if (gate === 0) continue;
    let h2 = (hue + gate * deg) % 180; if (h2 < 0) h2 += 180;   // H만 회전
    // hsv2rgb (OpenCV): H 0–179, S 0–255, V 0–255
    const sf = s / 255, hh = (h2 * 2) / 60;
    const ii = Math.floor(hh), f = hh - ii;
    const p = v * (1 - sf), q = v * (1 - sf * f), t = v * (1 - sf * (1 - f));
    let nr = 0, ng = 0, nb = 0;
    switch (((ii % 6) + 6) % 6) {
      case 0: nr = v; ng = t; nb = p; break;
      case 1: nr = q; ng = v; nb = p; break;
      case 2: nr = p; ng = v; nb = t; break;
      case 3: nr = p; ng = q; nb = v; break;
      case 4: nr = t; ng = p; nb = v; break;
      default: nr = v; ng = p; nb = q; break;
    }
    data[i] = nr; data[i + 1] = ng; data[i + 2] = nb;
  }
}

export default function VideoCorrection() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);

  const [cvdType, setCvdType] = useState<CVDType>("t");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [recording, setRecording] = useState(false);

  const clearRaf = () => { if (rafRef.current != null) cancelAnimationFrame(rafRef.current); rafRef.current = null; };

  const reset = () => {
    clearRaf();
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(null); setErrorMsg(""); setRecording(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const onFile = (file: File) => {
    if (!file.type.startsWith("video/")) { setErrorMsg("영상 파일(mp4, mov, webm)을 선택해주세요."); return; }
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setErrorMsg("");
    setVideoUrl(URL.createObjectURL(file));
  };
  const onDrop = (e: React.DragEvent) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) onFile(f); };

  // 실시간 보정 루프: 원본 <video>의 현재 프레임을 canvas에 그리고 t면 픽셀 보정.
  useEffect(() => {
    if (!videoUrl) return;
    const video = videoRef.current, canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    const draw = () => {
      const vw = video.videoWidth, vh = video.videoHeight;
      if (vw && vh) {
        const scale = Math.min(1, MAX_DIM / Math.max(vw, vh));
        const w = Math.round(vw * scale), h = Math.round(vh * scale);
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
        ctx.drawImage(video, 0, 0, w, h);
        if (cvdType === "t") {
          const img = ctx.getImageData(0, 0, w, h);
          correctTritan(img.data);
          ctx.putImageData(img, 0, 0);
        }
      }
      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);
    return clearRaf;
  }, [videoUrl, cvdType]);

  // 다운로드: canvas 스트림을 한 바퀴 녹화 (webm).
  const download = useCallback(async () => {
    const video = videoRef.current, canvas = canvasRef.current;
    if (!video || !canvas) return;
    try {
      setRecording(true);
      const stream = (canvas as HTMLCanvasElement & { captureStream(fps?: number): MediaStream }).captureStream(30);
      const rec = new MediaRecorder(stream, { mimeType: "video/webm" });
      const chunks: BlobPart[] = [];
      rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      const done = new Promise<void>((resolve) => { rec.onstop = () => resolve(); });
      const wasLoop = video.loop;
      video.loop = false; video.currentTime = 0;
      await video.play();
      rec.start();
      await new Promise<void>((resolve) => { video.onended = () => resolve(); });
      rec.stop();
      await done;
      video.loop = wasLoop; video.play();
      const blob = new Blob(chunks, { type: "video/webm" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `cvdlens_${cvdType}_corrected.webm`;
      a.click(); URL.revokeObjectURL(a.href);
    } catch {
      setErrorMsg("녹화가 지원되지 않는 브라우저입니다. 화면 녹화를 이용해주세요.");
    } finally { setRecording(false); }
  }, [cvdType]);

  return (
    <div className="flex flex-col items-center gap-6">
      {/* CVD 타입 선택 */}
      <div className="flex gap-2 flex-wrap justify-center">
        {(Object.keys(CVD_LABELS) as CVDType[]).map((type) => (
          <button key={type} onClick={() => setCvdType(type)}
            className="px-4 py-2 rounded-full text-sm font-medium transition-colors"
            style={{
              background: cvdType === type ? "var(--color-brand)" : "var(--bg-muted)",
              color: cvdType === type ? "#ffffff" : "var(--fg-muted)",
              border: "1px solid", borderColor: cvdType === type ? "var(--color-brand)" : "var(--border)",
            }}>
            {CVD_LABELS[type]}
          </button>
        ))}
      </div>

      {!videoUrl && (
        <div onDrop={onDrop} onDragOver={(e) => e.preventDefault()} onClick={() => fileInputRef.current?.click()}
          className="w-full max-w-md h-48 border-2 border-dashed rounded-xl flex flex-col items-center justify-center gap-2 cursor-pointer transition-colors"
          style={{ borderColor: "var(--border-strong)", color: "var(--fg-subtle)" }}
          onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--color-brand)")}
          onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border-strong)")}>
          <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M15 10l4.553-2.069A1 1 0 0121 8.868v6.264a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
          </svg>
          <p className="text-sm" style={{ color: "var(--fg-muted)" }}>영상을 드래그하거나 클릭해서 업로드</p>
          <p className="text-xs" style={{ color: "var(--fg-subtle)" }}>브라우저에서 실시간 보정 · 업로드 없음</p>
          <input ref={fileInputRef} type="file" accept="video/*" className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
        </div>
      )}

      {errorMsg && <p className="text-sm" style={{ color: "#d5383a" }}>{errorMsg}</p>}

      {videoUrl && (
        <div className="w-full flex flex-col items-center gap-5">
          <p className="text-xs font-mono" style={{ color: "var(--fg-subtle)" }}>브라우저 실시간 보정 · 서버 처리 없음</p>
          <div className="w-full grid grid-cols-2 gap-3 max-w-2xl">
            <div className="flex flex-col gap-1.5">
              <p className="text-xs font-mono text-center" style={{ color: "var(--fg-subtle)" }}>원본</p>
              <video ref={videoRef} src={videoUrl}
                className="w-full rounded-xl border" style={{ borderColor: "var(--border)", aspectRatio: "1/1", objectFit: "cover" }}
                controls loop muted autoPlay playsInline />
            </div>
            <div className="flex flex-col gap-1.5">
              <p className="text-xs font-mono text-center" style={{ color: "var(--color-brand)" }}>
                보정 · {CVD_LABELS[cvdType].split(" ")[0]}
              </p>
              <div className="relative w-full rounded-xl border overflow-hidden"
                style={{ borderColor: "var(--color-brand)", aspectRatio: "1/1", background: "var(--bg-muted)" }}>
                <canvas ref={canvasRef} className="w-full h-full" style={{ objectFit: "cover" }} />
                {cvdType !== "t" && (
                  <div className="absolute inset-0 flex items-center justify-center text-center p-4"
                    style={{ background: "rgba(0,0,0,0.5)", color: "#fff" }}>
                    <p className="text-sm">적색맹·녹색맹 실시간 영상 보정은 준비 중입니다.<br />이미지·카메라 탭에서 이용해주세요.</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="flex gap-3">
            {cvdType === "t" && (
              <button onClick={download} disabled={recording}
                className="px-5 py-2.5 rounded-full text-sm font-medium text-white transition-colors disabled:opacity-50"
                style={{ background: "var(--color-brand)" }}>
                {recording ? "녹화 중..." : "보정 영상 다운로드"}
              </button>
            )}
            <button onClick={reset}
              className="px-5 py-2.5 rounded-full text-sm font-medium transition-colors"
              style={{ background: "var(--bg-muted)", color: "var(--fg-muted)", border: "1px solid var(--border)" }}>
              다른 영상 선택
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
