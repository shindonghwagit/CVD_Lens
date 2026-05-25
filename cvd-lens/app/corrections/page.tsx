"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";

interface CorrectionRecord {
  id: string;
  cvd_type: string;
  source: string;
  original_image: string | null;
  corrected_image: string | null;
  created_at: string;
}

const CVD_LABEL: Record<string, { label: string; color: string }> = {
  p: { label: "적색맹", color: "#ef4444" },
  d: { label: "녹색맹", color: "#f97316" },
  t: { label: "청색맹", color: "#3b82f6" },
};

function cvdInfo(type: string) {
  return CVD_LABEL[type?.toLowerCase()] ?? { label: type ?? "알 수 없음", color: "var(--fg-subtle)" };
}

function formatDate(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString("ko-KR", { year: "numeric", month: "long", day: "numeric" })
    + " " + d.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
}

function CorrectionCard({ record }: { record: CorrectionRecord }) {
  const [showCorrected, setShowCorrected] = useState(false);
  const info = cvdInfo(record.cvd_type);
  const hasImages = record.original_image && record.corrected_image;

  return (
    <div
      className="rounded-[16px] border overflow-hidden"
      style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}
    >
      {hasImages ? (
        <div
          className="relative aspect-square cursor-pointer select-none"
          onClick={() => setShowCorrected((v) => !v)}
        >
          <img
            src={showCorrected ? record.corrected_image! : record.original_image!}
            alt={showCorrected ? "보정" : "원본"}
            className="w-full h-full object-cover"
            draggable={false}
          />
          <span
            className="absolute top-2 left-2 text-[11px] px-2 py-0.5 rounded-full text-white font-mono"
            style={{ background: showCorrected ? "var(--color-brand)" : "rgba(0,0,0,0.55)" }}
          >
            {showCorrected ? "보정" : "원본"}
          </span>
          <span className="absolute bottom-2 right-2 text-[10px] text-white/60 font-mono">탭하여 전환</span>
        </div>
      ) : (
        <div
          className="aspect-square flex items-center justify-center"
          style={{ background: "var(--bg-muted)" }}
        >
          <p className="text-xs font-mono" style={{ color: "var(--fg-subtle)" }}>이미지 없음</p>
        </div>
      )}

      <div className="px-3 py-2.5 flex items-center justify-between gap-2">
        <span
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono"
          style={{ background: info.color + "18", color: info.color }}
        >
          <span className="w-1.5 h-1.5 rounded-full inline-block" style={{ background: info.color }} />
          {info.label}
        </span>
        <span className="text-[11px] font-mono" style={{ color: "var(--fg-subtle)" }}>
          {formatDate(record.created_at)}
        </span>
      </div>
    </div>
  );
}

export default function CorrectionsPage() {
  const { status } = useSession();
  const router = useRouter();
  const [records, setRecords] = useState<CorrectionRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (status === "unauthenticated") { router.replace("/login"); return; }
    if (status !== "authenticated") return;

    fetch("/api/corrections")
      .then((r) => r.json())
      .then((data) => setRecords(Array.isArray(data) ? data : []))
      .finally(() => setLoading(false));
  }, [status, router]);

  return (
    <div className="max-w-[1280px] mx-auto px-6 md:px-10 py-10 md:py-14 pb-32">
      <div className="flex items-center gap-2 mb-8 font-mono text-[11px] tracking-[0.08em]" style={{ color: "var(--fg-subtle)" }}>
        <Link href="/" className="hover:text-fg transition-colors">HOME</Link>
        <span>/</span>
        <span style={{ color: "var(--fg)" }}>CORRECTIONS</span>
      </div>

      <div className="grid md:grid-cols-[1.4fr_1fr] gap-8 md:gap-10 items-end mb-10">
        <div>
          <p className="font-mono text-[11px] tracking-[0.16em] uppercase mb-3" style={{ color: "var(--fg-subtle)" }}>
            CORRECTION HISTORY
          </p>
          <h1 className="font-serif text-[36px] md:text-[60px] leading-[1.05] tracking-[-0.03em]">
            보정<br />기록
          </h1>
        </div>
        <p className="text-[15px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>
          저장한 색각 보정 이미지 목록입니다. 카드를 탭하면 원본과 보정 이미지를 전환할 수 있습니다.
        </p>
      </div>

      {loading ? (
        <div className="flex justify-center py-24">
          <div className="w-6 h-6 rounded-full border-2 border-brand border-t-transparent animate-spin" />
        </div>
      ) : records.length === 0 ? (
        <div
          className="rounded-[20px] border flex flex-col items-center justify-center py-24 gap-4"
          style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}
        >
          <p className="text-[15px]" style={{ color: "var(--fg-muted)" }}>저장된 보정 기록이 없습니다.</p>
          <Link
            href="/correction?tab=image"
            className="px-5 py-2 rounded-full text-[13px] text-white bg-brand hover:bg-brand-ink transition-colors"
          >
            이미지 보정하기
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
          {records.map((r) => (
            <CorrectionCard key={r.id} record={r} />
          ))}
        </div>
      )}

      <div className="mt-8 flex justify-end">
        <Link
          href="/correction?tab=image"
          className="px-5 py-2 rounded-full text-[13px] border transition-colors hover:bg-[var(--bg-muted)]"
          style={{ borderColor: "var(--border-strong)", color: "var(--fg)" }}
        >
          이미지 보정하기 →
        </Link>
      </div>
    </div>
  );
}
