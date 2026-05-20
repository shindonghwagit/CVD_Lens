"use client";

import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import { Suspense } from "react";
import CameraView from "../components/CameraView";
import ImageCorrection from "../components/ImageCorrection";

function CorrectionContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const tab = searchParams.get("tab") === "image" ? "image" : "camera";

  const isImage = tab === "image";

  const setTab = (next: "camera" | "image") => {
    const params = new URLSearchParams(searchParams.toString());
    if (next === "image") params.set("tab", "image");
    else params.delete("tab");
    router.replace(`/correction${params.toString() ? "?" + params.toString() : ""}`);
  };

  return (
    <div className="max-w-[1280px] mx-auto px-6 md:px-10 py-10 md:py-14 pb-32">
      {/* Breadcrumbs */}
      <div className="flex items-center gap-2 mb-8 font-mono text-[11px] tracking-[0.08em]" style={{ color: "var(--fg-subtle)" }}>
        <Link href="/" className="hover:text-fg transition-colors">HOME</Link>
        <span>/</span>
        <span style={{ color: "var(--fg)" }}>
          {isImage ? "02 · IMAGE" : "03 · CAMERA"}
        </span>
      </div>

      {/* Head */}
      <div className="grid md:grid-cols-[1.4fr_1fr] gap-8 md:gap-10 items-end mb-10">
        <div>
          <p className="font-mono text-[11px] tracking-[0.16em] uppercase mb-3" style={{ color: "var(--fg-subtle)" }}>
            {isImage ? "UPLOAD · CORRECT · COMPARE" : "CAMERA · CAPTURE · CORRECT"}
          </p>
          <h1 className="font-serif text-[36px] md:text-[60px] leading-[1.05] tracking-[-0.03em]">
            {isImage ? (
              <>이미지 파일<br />AI 색 보정</>
            ) : (
              <>카메라 촬영<br />AI 색 보정</>
            )}
          </h1>
        </div>
        <p className="text-[15px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>
          {isImage
            ? "사진을 업로드하면 선택한 색각이상 타입에 맞춰 색 대비를 재구성합니다. 원본과 보정본은 슬라이더로 직관적으로 비교할 수 있습니다."
            : "카메라로 사진을 촬영하면 AI 서버가 색각이상 타입에 맞춰 색 대비를 보정합니다. 원본과 보정본을 슬라이더로 비교할 수 있습니다."}
        </p>
      </div>

      {/* Tab switcher */}
      <div className="mb-6 inline-flex p-1 rounded-full border" style={{ background: "var(--bg-muted)", borderColor: "var(--border)" }}>
        <button
          onClick={() => setTab("camera")}
          className="px-4 py-2 rounded-full text-[12px] font-mono tracking-[0.06em] transition-colors"
          style={{
            background: !isImage ? "var(--bg-elevated)" : "transparent",
            color: !isImage ? "var(--fg)" : "var(--fg-muted)",
            boxShadow: !isImage ? "var(--shadow-soft)" : "none",
          }}
        >
          CAMERA
        </button>
        <button
          onClick={() => setTab("image")}
          className="px-4 py-2 rounded-full text-[12px] font-mono tracking-[0.06em] transition-colors"
          style={{
            background: isImage ? "var(--bg-elevated)" : "transparent",
            color: isImage ? "var(--fg)" : "var(--fg-muted)",
            boxShadow: isImage ? "var(--shadow-soft)" : "none",
          }}
        >
          IMAGE FILE
        </button>
      </div>

      {/* Content panel */}
      <div
        className="rounded-[20px] border p-6 md:p-8 shadow-soft"
        style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}
      >
        {isImage ? <ImageCorrection /> : <CameraView />}
      </div>

    </div>
  );
}

export default function CorrectionPage() {
  return (
    <Suspense fallback={<div className="max-w-[1280px] mx-auto px-6 md:px-10 py-14" style={{ color: "var(--fg-muted)" }}>Loading...</div>}>
      <CorrectionContent />
    </Suspense>
  );
}