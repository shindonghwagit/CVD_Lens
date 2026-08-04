"use client";

import Link from "next/link";
import IshiharaTest from "../components/IshiharaTest";

export default function IshiharaPage() {
  return (
    <div className="max-w-[1280px] mx-auto px-6 md:px-10 py-10 md:py-14 pb-32">
      {/* Breadcrumbs */}
      <div className="flex items-center gap-2 mb-8 font-mono text-[11px] tracking-[0.08em]" style={{ color: "var(--fg-subtle)" }}>
        <Link href="/" className="hover:text-fg transition-colors">HOME</Link>
        <span>/</span>
        <span style={{ color: "var(--fg)" }}>01 · ASSESSMENT</span>
      </div>

      {/* Head */}
      <div className="grid md:grid-cols-[1.4fr_1fr] gap-8 md:gap-10 items-end mb-10">
        <div>
          <p className="font-mono text-[11px] tracking-[0.16em] uppercase mb-3" style={{ color: "var(--fg-subtle)" }}>
            GUIDED SELF-TEST
          </p>
          <h1 className="font-serif text-[36px] md:text-[60px] leading-[1.05] tracking-[-0.03em]">
            이시하라
            <br />색각 진단
          </h1>
        </div>
        <p className="text-[15px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>
          아래 플레이트에 숨어있는 숫자를 입력해주세요. 진단은 보정 없이 진행되며,
          결과 화면에서 틀린 도판을 AI 보정으로 다시 볼 수 있습니다. 적록 계열(적색맹·녹색맹)
          스크리닝이며 총 5장 · 약 2분이 소요됩니다.
        </p>
      </div>

      {/* Test panel */}
      <div
        className="rounded-[20px] border p-8 md:p-10 shadow-soft"
        style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}
      >
        <IshiharaTest />
      </div>

      {/* Disclaimer */}
      <p className="mt-6 text-xs font-mono tracking-[0.04em]" style={{ color: "var(--fg-subtle)" }}>
        ※ 본 검사는 적록 계열(적색맹·녹색맹) 스크리닝이며 청색 계열(청색맹) 감별은 포함하지 않습니다.
        참고용 자가 진단으로 의학적 진단을 대체하지 않으니, 정밀 검사는 안과 전문의에게 문의하세요.
      </p>
    </div>
  );
}