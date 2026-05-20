"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";

interface Result {
  id: string;
  correct: number;
  total: number;
  diagnosis: string;
  created_at: string;
}

const DIAGNOSIS_LABEL: Record<string, { label: string; color: string }> = {
  normal:       { label: "정상",         color: "#22c55e" },
  p:            { label: "제1색각이상",   color: "#ef4444" },
  d:            { label: "제2색각이상",   color: "#f97316" },
  t:            { label: "제3색각이상",   color: "#3b82f6" },
  protanopia:   { label: "제1색각이상",   color: "#ef4444" },
  deuteranopia: { label: "제2색각이상",   color: "#f97316" },
  tritanopia:   { label: "제3색각이상",   color: "#3b82f6" },
};

function diagnosisInfo(diagnosis: string) {
  const key = diagnosis?.toLowerCase();
  return DIAGNOSIS_LABEL[key] ?? { label: diagnosis ?? "알 수 없음", color: "var(--fg-subtle)" };
}

function formatDate(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString("ko-KR", { year: "numeric", month: "long", day: "numeric" })
    + " "
    + d.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
}

export default function HistoryPage() {
  const { status } = useSession();
  const router = useRouter();
  const [results, setResults] = useState<Result[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (status === "unauthenticated") {
      router.replace("/login");
      return;
    }
    if (status !== "authenticated") return;

    fetch("/api/results")
      .then((r) => r.json())
      .then((data) => setResults(Array.isArray(data) ? data : []))
      .finally(() => setLoading(false));
  }, [status, router]);

  return (
    <div className="max-w-[1280px] mx-auto px-6 md:px-10 py-10 md:py-14 pb-32">
      {/* Breadcrumbs */}
      <div className="flex items-center gap-2 mb-8 font-mono text-[11px] tracking-[0.08em]" style={{ color: "var(--fg-subtle)" }}>
        <Link href="/" className="hover:text-fg transition-colors">HOME</Link>
        <span>/</span>
        <span style={{ color: "var(--fg)" }}>HISTORY</span>
      </div>

      {/* Head */}
      <div className="grid md:grid-cols-[1.4fr_1fr] gap-8 md:gap-10 items-end mb-10">
        <div>
          <p className="font-mono text-[11px] tracking-[0.16em] uppercase mb-3" style={{ color: "var(--fg-subtle)" }}>
            DIAGNOSIS RECORDS
          </p>
          <h1 className="font-serif text-[36px] md:text-[60px] leading-[1.05] tracking-[-0.03em]">
            진단
            <br />기록
          </h1>
        </div>
        <p className="text-[15px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>
          이시하라 색각 진단 결과 기록입니다. 최근 10회까지 표시됩니다.
        </p>
      </div>

      {/* Content */}
      <div
        className="rounded-[20px] border shadow-soft overflow-hidden"
        style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}
      >
        {loading ? (
          <div className="flex items-center justify-center py-24">
            <div className="w-6 h-6 rounded-full border-2 border-brand border-t-transparent animate-spin" />
          </div>
        ) : results.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 gap-4">
            <p className="text-[15px]" style={{ color: "var(--fg-muted)" }}>아직 진단 기록이 없습니다.</p>
            <Link
              href="/ishihara"
              className="px-5 py-2 rounded-full text-[13px] text-white bg-brand hover:bg-brand-ink transition-colors"
            >
              진단 시작하기
            </Link>
          </div>
        ) : (
          <table className="w-full text-[14px]">
            <thead>
              <tr className="border-b font-mono text-[11px] tracking-[0.08em] uppercase" style={{ borderColor: "var(--border)", color: "var(--fg-subtle)" }}>
                <th className="text-left px-6 md:px-8 py-4">일시</th>
                <th className="text-left px-4 py-4">진단 결과</th>
                <th className="text-left px-4 py-4">정답률</th>
                <th className="text-left px-4 py-4 hidden md:table-cell">정답 / 전체</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => {
                const info = diagnosisInfo(r.diagnosis);
                const pct = Math.round((r.correct / r.total) * 100);
                return (
                  <tr
                    key={r.id}
                    className="border-b last:border-0 transition-colors hover:bg-[var(--bg-muted)]"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <td className="px-6 md:px-8 py-4 font-mono text-[12px]" style={{ color: "var(--fg-muted)" }}>
                      <div className="flex items-center gap-3">
                        <span
                          className="w-5 h-5 flex items-center justify-center rounded-full text-[10px] font-mono shrink-0"
                          style={{ background: "var(--bg-muted)", color: "var(--fg-subtle)" }}
                        >
                          {i + 1}
                        </span>
                        {formatDate(r.created_at)}
                      </div>
                    </td>
                    <td className="px-4 py-4">
                      <span
                        className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[12px] font-mono"
                        style={{ background: info.color + "18", color: info.color }}
                      >
                        <span className="w-1.5 h-1.5 rounded-full inline-block" style={{ background: info.color }} />
                        {info.label}
                      </span>
                    </td>
                    <td className="px-4 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-20 h-1.5 rounded-full overflow-hidden" style={{ background: "var(--bg-muted)" }}>
                          <div
                            className="h-full rounded-full transition-all"
                            style={{ width: `${pct}%`, background: info.color }}
                          />
                        </div>
                        <span className="font-mono text-[12px]" style={{ color: "var(--fg-muted)" }}>{pct}%</span>
                      </div>
                    </td>
                    <td className="px-4 py-4 font-mono text-[12px] hidden md:table-cell" style={{ color: "var(--fg-muted)" }}>
                      {r.correct} / {r.total}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <div className="mt-6 flex justify-end">
        <Link
          href="/ishihara"
          className="px-5 py-2 rounded-full text-[13px] border transition-colors hover:bg-[var(--bg-muted)]"
          style={{ borderColor: "var(--border-strong)", color: "var(--fg)" }}
        >
          다시 진단하기 →
        </Link>
      </div>
    </div>
  );
}
