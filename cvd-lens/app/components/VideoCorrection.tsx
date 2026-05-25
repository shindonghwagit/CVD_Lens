"use client";

import { useCallback, useRef, useState } from "react";
import { CVDType } from "../hooks/useCVDModel";

const CVD_LABELS: Record<CVDType, string> = {
  p: "적색맹 (Protanopia)",
  d: "녹색맹 (Deuteranopia)",
  t: "청색맹 (Tritanopia)",
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Stage = "idle" | "uploading" | "processing" | "done" | "error";

export default function VideoCorrection() {
  const fileInputRef  = useRef<HTMLInputElement>(null);
  const origVideoRef  = useRef<HTMLVideoElement>(null);
  const corrVideoRef  = useRef<HTMLVideoElement>(null);

  const [cvdType, setCvdType]   = useState<CVDType>("d");
  const [stage, setStage]       = useState<Stage>("idle");
  const [progress, setProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");
  const [origUrl, setOrigUrl]   = useState<string | null>(null);
  const [corrUrl, setCorrUrl]   = useState<string | null>(null);
  const [frameCount, setFrameCount] = useState<number | null>(null);
  const [elapsed, setElapsed]   = useState<number | null>(null);

  const reset = () => {
    if (origUrl) URL.revokeObjectURL(origUrl);
    if (corrUrl) URL.revokeObjectURL(corrUrl);
    setOrigUrl(null); setCorrUrl(null);
    setStage("idle"); setProgress(0);
    setFrameCount(null); setElapsed(null); setErrorMsg("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const processVideo = useCallback(async (file: File) => {
    if (origUrl) URL.revokeObjectURL(origUrl);
    if (corrUrl) URL.revokeObjectURL(corrUrl);
    setCorrUrl(null); setErrorMsg("");

    const localUrl = URL.createObjectURL(file);
    setOrigUrl(localUrl);
    setStage("uploading");
    setProgress(10);

    const form = new FormData();
    form.append("video", file);
    form.append("cvd_type", cvdType);

    const t0 = Date.now();

    try {
      setStage("processing");
      setProgress(30);

      // simulate progress while waiting
      const progTimer = setInterval(() => {
        setProgress((p) => Math.min(p + 2, 88));
      }, 800);

      const res = await fetch(`${API_URL}/infer/video`, { method: "POST", body: form });

      clearInterval(progTimer);

      if (!res.ok) {
        const json = await res.json().catch(() => ({}));
        throw new Error(json.error ?? `서버 오류 ${res.status}`);
      }

      const frames = Number(res.headers.get("X-Frame-Count") ?? 0);
      setFrameCount(frames || null);
      setElapsed(Math.round((Date.now() - t0) / 1000));

      const blob = await res.blob();
      setCorrUrl(URL.createObjectURL(blob));
      setProgress(100);
      setStage("done");
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "처리 중 오류가 발생했습니다.");
      setStage("error");
    }
  }, [cvdType, origUrl, corrUrl]);

  const onFile = (file: File) => {
    if (!file.type.startsWith("video/")) {
      setErrorMsg("영상 파일(mp4, mov, avi)을 선택해주세요."); setStage("error"); return;
    }
    processVideo(file);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  };

  const isBusy = stage === "uploading" || stage === "processing";

  return (
    <div className="flex flex-col items-center gap-6">
      {/* CVD 타입 선택 */}
      <div className="flex gap-2 flex-wrap justify-center">
        {(Object.keys(CVD_LABELS) as CVDType[]).map((type) => (
          <button
            key={type}
            onClick={() => setCvdType(type)}
            disabled={isBusy}
            className="px-4 py-2 rounded-full text-sm font-medium transition-colors disabled:opacity-40"
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

      {/* Upload area */}
      {stage === "idle" && (
        <div
          onDrop={onDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => fileInputRef.current?.click()}
          className="w-full max-w-md h-48 border-2 border-dashed rounded-xl flex flex-col items-center justify-center gap-2 cursor-pointer transition-colors"
          style={{ borderColor: "var(--border-strong)", color: "var(--fg-subtle)" }}
          onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--color-brand)")}
          onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border-strong)")}
        >
          <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M15 10l4.553-2.069A1 1 0 0121 8.868v6.264a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
          </svg>
          <p className="text-sm" style={{ color: "var(--fg-muted)" }}>영상을 드래그하거나 클릭해서 업로드</p>
          <p className="text-xs" style={{ color: "var(--fg-subtle)" }}>MP4 · MOV · AVI · 권장 30초 이하</p>
          <input ref={fileInputRef} type="file" accept="video/*" className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
        </div>
      )}

      {/* Processing */}
      {isBusy && (
        <div className="w-full max-w-md flex flex-col gap-3">
          <div className="flex justify-between text-sm" style={{ color: "var(--fg-muted)" }}>
            <span>{stage === "uploading" ? "업로드 중..." : "프레임 보정 중..."}</span>
            <span>{progress}%</span>
          </div>
          <div className="w-full h-2 rounded-full overflow-hidden" style={{ background: "var(--bg-muted)" }}>
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${progress}%`, background: "var(--color-brand)" }}
            />
          </div>
          <p className="text-xs text-center" style={{ color: "var(--fg-subtle)" }}>
            영상 길이에 따라 수십 초~수 분 소요될 수 있습니다
          </p>
        </div>
      )}

      {/* Error */}
      {stage === "error" && (
        <div className="w-full max-w-md flex flex-col items-center gap-3">
          <p className="text-sm" style={{ color: "#d5383a" }}>{errorMsg}</p>
          <button onClick={reset} className="px-5 py-2 rounded-full text-sm font-medium transition-colors"
            style={{ background: "var(--bg-muted)", color: "var(--fg)", border: "1px solid var(--border)" }}>
            다시 시도
          </button>
        </div>
      )}

      {/* Result */}
      {stage === "done" && origUrl && corrUrl && (
        <div className="w-full flex flex-col items-center gap-5">
          {/* Stats */}
          {(frameCount || elapsed) && (
            <div className="flex gap-4 text-xs font-mono" style={{ color: "var(--fg-subtle)" }}>
              {frameCount && <span>{frameCount} frames</span>}
              {elapsed && <span>처리 시간 {elapsed}s</span>}
            </div>
          )}

          {/* Side-by-side videos */}
          <div className="w-full grid grid-cols-2 gap-3 max-w-2xl">
            <div className="flex flex-col gap-1.5">
              <p className="text-xs font-mono text-center" style={{ color: "var(--fg-subtle)" }}>원본</p>
              <video
                ref={origVideoRef}
                src={origUrl}
                className="w-full rounded-xl border object-cover"
                style={{ borderColor: "var(--border)", aspectRatio: "1/1" }}
                controls
                loop
                muted
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <p className="text-xs font-mono text-center" style={{ color: "var(--color-brand)" }}>
                보정 · {CVD_LABELS[cvdType].split(" ")[0]}
              </p>
              <video
                ref={corrVideoRef}
                src={corrUrl}
                className="w-full rounded-xl border object-cover"
                style={{ borderColor: "var(--color-brand)", aspectRatio: "1/1" }}
                controls
                loop
                muted
              />
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <a
              href={corrUrl}
              download={`cvdlens_${cvdType}_corrected.mp4`}
              className="px-5 py-2.5 rounded-full text-sm font-medium text-white transition-colors"
              style={{ background: "var(--color-brand)" }}
            >
              보정 영상 다운로드
            </a>
            <button
              onClick={reset}
              className="px-5 py-2.5 rounded-full text-sm font-medium transition-colors"
              style={{ background: "var(--bg-muted)", color: "var(--fg-muted)", border: "1px solid var(--border)" }}
            >
              다른 영상 선택
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
