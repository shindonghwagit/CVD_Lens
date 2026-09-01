"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CVDType } from "../hooks/useCVDModel";
import { preloadSession, runOnnxCorrection, ONNX_SIZE } from "../../lib/cvdOnnx";

const CVD_LABELS: Record<CVDType, string> = {
  p: "적색맹 (Protanopia)",
  d: "녹색맹 (Deuteranopia)",
  t: "청색맹 (Tritanopia)",
};

const MAX_DIM = 512;          // 실시간 캔버스 최대 변
const TRITAN_DEG = 30;        // 청색맹 hue 회전각 (배포 _tritan_hue_shift와 동일, severity 1.0)
const PD_SEVERITY = 0.7;      // 적/녹색맹 기본 강도 (이미지 경로 기본값과 동일)

/** 청색맹(t) 채도보존 hue 회전 — 순수 canvas 픽셀 연산(모델 불필요). */
function correctTritan(data: Uint8ClampedArray) {
  const deg = TRITAN_DEG;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const v = Math.max(r, g, b), mn = Math.min(r, g, b), diff = v - mn;
    if (diff === 0) continue;
    const s = (diff * 255) / v;
    let hue: number;
    if (v === r) hue = 60 * (g - b) / diff;
    else if (v === g) hue = 120 + 60 * (b - r) / diff;
    else hue = 240 + 60 * (r - g) / diff;
    hue = hue / 2; if (hue < 0) hue += 180;
    const satG = Math.min(Math.max((s - 18) / 50, 0), 1);
    if (satG === 0) continue;
    const band = (x: number, lo: number, hi: number) =>
      Math.min(Math.max((x - lo) / 12, 0), 1) * Math.min(Math.max((hi - x) / 12, 0), 1);
    const gate = satG * Math.max(band(hue, 90, 135), band(hue, 18, 40));
    if (gate === 0) continue;
    let h2 = (hue + gate * deg) % 180; if (h2 < 0) h2 += 180;
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

type ModelStatus = "idle" | "loading" | "ready" | "error";

export default function VideoCorrection() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const modelReadyRef = useRef(false);

  const [cvdType, setCvdType] = useState<CVDType>("t");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [recording, setRecording] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus>("idle");
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved">("idle");

  const clearRaf = () => { if (rafRef.current != null) cancelAnimationFrame(rafRef.current); rafRef.current = null; };

  const reset = () => {
    clearRaf();
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(null); setErrorMsg(""); setRecording(false); setSaveState("idle");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const onFile = (file: File) => {
    if (!file.type.startsWith("video/")) { setErrorMsg("영상 파일(mp4, mov, webm)을 선택해주세요."); return; }
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setErrorMsg("");
    setVideoUrl(URL.createObjectURL(file));
  };
  const onDrop = (e: React.DragEvent) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) onFile(f); };

  // 실시간 보정 루프: t=canvas hue 회전(동기), p/d=ONNX 추론(비동기)+delta 업샘플 합성.
  useEffect(() => {
    if (!videoUrl) return;
    const video = videoRef.current, canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    setSaveState("idle");   // 타입/영상 바뀌면 저장 상태 초기화

    const in256 = document.createElement("canvas"); in256.width = ONNX_SIZE; in256.height = ONNX_SIZE;
    const in256ctx = in256.getContext("2d", { willReadFrequently: true })!;
    const dEnc = document.createElement("canvas"); dEnc.width = ONNX_SIZE; dEnc.height = ONNX_SIZE;
    const dEncCtx = dEnc.getContext("2d", { willReadFrequently: true })!;
    const dUp = document.createElement("canvas");
    const dUpCtx = dUp.getContext("2d", { willReadFrequently: true })!;

    let cancelled = false;
    let busy = false;

    modelReadyRef.current = false;
    if (cvdType !== "t") {
      setModelStatus("loading");
      preloadSession(cvdType)
        .then(() => { if (!cancelled) { modelReadyRef.current = true; setModelStatus("ready"); } })
        .catch(() => { if (!cancelled) setModelStatus("error"); });
    } else {
      setModelStatus("idle");
    }

    const sizeTo = () => {
      const vw = video.videoWidth, vh = video.videoHeight;
      if (!vw || !vh) return null;
      const scale = Math.min(1, MAX_DIM / Math.max(vw, vh));
      const w = Math.round(vw * scale), h = Math.round(vh * scale);
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      return { w, h };
    };

    const runPd = (w: number, h: number) => {
      busy = true;
      in256ctx.drawImage(video, 0, 0, ONNX_SIZE, ONNX_SIZE);
      const src = in256ctx.getImageData(0, 0, ONNX_SIZE, ONNX_SIZE);
      runOnnxCorrection(cvdType, src, PD_SEVERITY).then((out) => {
        if (cancelled) { busy = false; return; }
        const N = ONNX_SIZE * ONNX_SIZE;
        const de = dEncCtx.createImageData(ONNX_SIZE, ONNX_SIZE);
        const sd = src.data;
        for (let i = 0; i < N; i++) {                       // delta = out - in, (delta+1)*127.5 로 인코딩
          de.data[i * 4] = (out[i] - sd[i * 4] / 255 + 1) * 127.5;
          de.data[i * 4 + 1] = (out[N + i] - sd[i * 4 + 1] / 255 + 1) * 127.5;
          de.data[i * 4 + 2] = (out[2 * N + i] - sd[i * 4 + 2] / 255 + 1) * 127.5;
          de.data[i * 4 + 3] = 255;
        }
        dEncCtx.putImageData(de, 0, 0);
        dUp.width = w; dUp.height = h; dUpCtx.imageSmoothingEnabled = true;
        dUpCtx.drawImage(dEnc, 0, 0, w, h);                 // delta 원해상도 업샘플(bilinear)
        const dUpImg = dUpCtx.getImageData(0, 0, w, h);
        ctx.drawImage(video, 0, 0, w, h);                   // 현재 원본 프레임
        const frame = ctx.getImageData(0, 0, w, h);
        const fd = frame.data, ud = dUpImg.data;
        for (let i = 0; i < fd.length; i += 4) {            // 합성: orig + delta
          fd[i] = fd[i] + (ud[i] / 127.5 - 1) * 255;
          fd[i + 1] = fd[i + 1] + (ud[i + 1] / 127.5 - 1) * 255;
          fd[i + 2] = fd[i + 2] + (ud[i + 2] / 127.5 - 1) * 255;
        }
        ctx.putImageData(frame, 0, 0);
        busy = false;
      }).catch(() => { if (!cancelled) setModelStatus("error"); busy = false; });
    };

    const loop = () => {
      if (cancelled) return;
      const dim = sizeTo();
      if (dim) {
        if (cvdType === "t") {
          ctx.drawImage(video, 0, 0, dim.w, dim.h);
          const img = ctx.getImageData(0, 0, dim.w, dim.h);
          correctTritan(img.data);
          ctx.putImageData(img, 0, 0);
        } else if (modelReadyRef.current) {
          if (!busy) runPd(dim.w, dim.h);                   // 이전 추론 끝났을 때만 새로 (비동기 스로틀)
        } else {
          ctx.drawImage(video, 0, 0, dim.w, dim.h);         // 모델 로딩 중엔 원본 표시
        }
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => { cancelled = true; clearRaf(); };
  }, [videoUrl, cvdType]);

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

  // 목록에 저장 — 영상 전체 대신 현재 대표 프레임(원본/보정) 썸네일 쌍을 저장.
  // /api/corrections 스키마(썸네일 base64)와 동일. source="video"로 구분.
  const saveCorrection = useCallback(async () => {
    const video = videoRef.current, canvas = canvasRef.current;
    if (!video || !canvas || saveState === "saving" || saveState === "saved") return;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh || !canvas.width) return;
    setSaveState("saving");

    const square = (draw: (c: CanvasRenderingContext2D) => void): string => {
      const c = document.createElement("canvas"); c.width = 256; c.height = 256;
      const cx = c.getContext("2d")!;
      draw(cx);
      return c.toDataURL("image/jpeg", 0.75);
    };
    // 원본: 현재 영상 프레임 center-crop
    const oSide = Math.min(vw, vh);
    const originalImage = square((cx) => cx.drawImage(video, (vw - oSide) / 2, (vh - oSide) / 2, oSide, oSide, 0, 0, 256, 256));
    // 보정: 현재 canvas(보정 출력) center-crop
    const cw = canvas.width, ch = canvas.height, cSide = Math.min(cw, ch);
    const correctedImage = square((cx) => cx.drawImage(canvas, (cw - cSide) / 2, (ch - cSide) / 2, cSide, cSide, 0, 0, 256, 256));

    try {
      const res = await fetch("/api/corrections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cvdType, source: "video", originalImage, correctedImage }),
      });
      if (res.ok) setSaveState("saved");
      else { setSaveState("idle"); if (res.status === 401) setErrorMsg("목록 저장은 로그인이 필요합니다."); }
    } catch {
      setSaveState("idle");
    }
  }, [cvdType, saveState]);

  const canDownload = cvdType === "t" || modelStatus === "ready";

  return (
    <div className="flex flex-col items-center gap-6">
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
          <p className="text-xs font-mono" style={{ color: "var(--fg-subtle)" }}>
            브라우저 실시간 보정 · 서버 처리 없음{cvdType !== "t" ? " · 학습 모델(ONNX Web)" : ""}
          </p>
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
                {cvdType !== "t" && modelStatus === "loading" && (
                  <div className="absolute inset-0 flex items-center justify-center"
                    style={{ background: "rgba(0,0,0,0.45)", color: "#fff" }}>
                    <p className="text-sm">모델 로딩 중... (첫 실행 시 잠시 소요)</p>
                  </div>
                )}
                {cvdType !== "t" && modelStatus === "error" && (
                  <div className="absolute inset-0 flex items-center justify-center text-center p-4"
                    style={{ background: "rgba(0,0,0,0.55)", color: "#fff" }}>
                    <p className="text-sm">모델을 불러오지 못했습니다. 새로고침 후 다시 시도해주세요.</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="flex gap-3">
            <button onClick={download} disabled={recording || !canDownload}
              className="px-5 py-2.5 rounded-full text-sm font-medium text-white transition-colors disabled:opacity-50"
              style={{ background: "var(--color-brand)" }}>
              {recording ? "녹화 중..." : "보정 영상 다운로드"}
            </button>
            <button onClick={saveCorrection} disabled={!canDownload || saveState === "saving" || saveState === "saved"}
              className="px-5 py-2.5 rounded-full text-sm font-medium transition-colors disabled:opacity-60"
              style={saveState === "saved"
                ? { background: "#22c55e18", color: "#22c55e", border: "1px solid #22c55e44" }
                : { background: "var(--bg-muted)", color: "var(--fg)", border: "1px solid var(--border-strong)" }
              }>
              {saveState === "saving" ? "저장 중..." : saveState === "saved" ? "저장됨 ✓" : "목록에 저장"}
            </button>
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
