"use client";

import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import { Suspense, useEffect } from "react";
import CameraView from "../components/CameraView";
import ImageCorrection from "../components/ImageCorrection";
import VideoCorrection from "../components/VideoCorrection";
import { CVDType } from "../hooks/useCVDModel";

type Tab = "camera" | "image" | "video";

function CorrectionContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const raw = searchParams.get("tab");
  const tab: Tab = raw === "image" ? "image" : raw === "video" ? "video" : "camera";

  // Optional ?type= from the ishihara result → default CVD type in the image tab.
  const typeRaw = searchParams.get("type");
  const initialType: CVDType | undefined =
    typeRaw === "p" || typeRaw === "d" || typeRaw === "t" ? typeRaw : undefined;

  // Pre-warm the inference server on mount: Render free instances sleep, so a
  // fire-and-forget /health ping starts waking it while the user reads the page.
  // Response is intentionally not awaited; failures are ignored.
  useEffect(() => {
    const api = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    fetch(`${api}/health`).catch(() => {});
  }, []);

  const setTab = (next: Tab) => {
    const params = new URLSearchParams(searchParams.toString());
    if (next === "camera") params.delete("tab");
    else params.set("tab", next);
    router.replace(`/correction${params.toString() ? "?" + params.toString() : ""}`);
  };

  const headings: Record<Tab, { eyebrow: string; title: React.ReactNode; desc: string }> = {
    camera: {
      eyebrow: "CAMERA · CAPTURE · CORRECT",
      title: <>카메라 촬영<br />AI 색 보정</>,
      desc: "카메라로 사진을 촬영하면 AI 서버가 색각이상 타입에 맞춰 색 대비를 보정합니다. 원본과 보정본을 슬라이더로 비교할 수 있습니다.",
    },
    image: {
      eyebrow: "UPLOAD · CORRECT · COMPARE",
      title: <>이미지 파일<br />AI 색 보정</>,
      desc: "사진을 업로드하면 선택한 색각이상 타입에 맞춰 색 대비를 재구성합니다. 원본과 보정본은 슬라이더로 직관적으로 비교할 수 있습니다.",
    },
    video: {
      eyebrow: "UPLOAD · PROCESS · DOWNLOAD",
      title: <>영상 파일<br />AI 색 보정</>,
      desc: "영상을 업로드하면 프레임 단위로 색각이상 보정을 적용합니다. 원본과 보정 영상을 나란히 비교하고 다운로드할 수 있습니다.",
    },
  };

  const h = headings[tab];

  const TABS: { key: Tab; label: string }[] = [
    { key: "camera", label: "CAMERA" },
    { key: "image",  label: "IMAGE FILE" },
    { key: "video",  label: "VIDEO FILE" },
  ];

  return (
    <div className="max-w-[1280px] mx-auto px-6 md:px-10 py-10 md:py-14 pb-32">
      {/* Breadcrumbs */}
      <div className="flex items-center gap-2 mb-8 font-mono text-[11px] tracking-[0.08em]" style={{ color: "var(--fg-subtle)" }}>
        <Link href="/" className="hover:text-fg transition-colors">HOME</Link>
        <span>/</span>
        <span style={{ color: "var(--fg)" }}>
          {tab === "image" ? "02 · IMAGE" : tab === "video" ? "04 · VIDEO" : "03 · CAMERA"}
        </span>
      </div>

      {/* Head */}
      <div className="grid md:grid-cols-[1.4fr_1fr] gap-8 md:gap-10 items-end mb-10">
        <div>
          <p className="font-mono text-[11px] tracking-[0.16em] uppercase mb-3" style={{ color: "var(--fg-subtle)" }}>
            {h.eyebrow}
          </p>
          <h1 className="font-serif text-[36px] md:text-[60px] leading-[1.05] tracking-[-0.03em]">
            {h.title}
          </h1>
        </div>
        <p className="text-[15px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>
          {h.desc}
        </p>
      </div>

      {/* Tab switcher */}
      <div className="mb-6 inline-flex p-1 rounded-full border" style={{ background: "var(--bg-muted)", borderColor: "var(--border)" }}>
        {TABS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className="px-4 py-2 rounded-full text-[12px] font-mono tracking-[0.06em] transition-colors"
            style={{
              background: tab === key ? "var(--bg-elevated)" : "transparent",
              color: tab === key ? "var(--fg)" : "var(--fg-muted)",
              boxShadow: tab === key ? "var(--shadow-soft)" : "none",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Content panel */}
      <div
        className="rounded-[20px] border p-6 md:p-8 shadow-soft"
        style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}
      >
        {tab === "image" ? <ImageCorrection initialType={initialType} /> : tab === "video" ? <VideoCorrection /> : <CameraView />}
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
