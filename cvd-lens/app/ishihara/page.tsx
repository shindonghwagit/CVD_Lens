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
            <br />간이 색각 선별
          </h1>
        </div>
        <p className="text-[15px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>
          아래 플레이트의 숫자를 입력해 적록 색각 반응을 간단히 확인합니다.
          결과는 적색맹·녹색맹 관련 반응 경향까지 보수적으로 안내하며,
          총 10장 · 약 3분이 소요됩니다.
        </p>
      </div>

      <div
        className="mb-5 rounded-xl border px-5 py-4 text-[13px] leading-relaxed"
        style={{ background: "var(--bg-muted)", borderColor: "var(--border)", color: "var(--fg-muted)" }}
      >
        <p className="font-medium mb-1" style={{ color: "var(--fg)" }}>검사 환경을 먼저 확인해 주세요</p>
        <p>야간 모드·색상 필터·색약 보정 기능을 끄고, 화면 밝기를 적당히 높인 뒤 밝고 눈부심 없는 곳에서 진행하세요.</p>
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
        ※ 본 검사는 적록 계열의 화면 기반 간이 선별검사이며 청황색 계열은 포함하지 않습니다.
        모니터와 주변 조명의 영향을 받으므로 의학적 진단을 대체하지 않습니다.
      </p>

      <div className="mt-3 text-[11px] leading-relaxed" style={{ color: "var(--fg-subtle)" }}>
        판독 기준 참고: {" "}
        <a className="underline underline-offset-2" href="https://www.hse.gov.uk/pubns/ms7.pdf" target="_blank" rel="noreferrer">영국 HSE 색각검사 지침</a>
        {" · "}
        <a className="underline underline-offset-2" href="https://pubmed.ncbi.nlm.nih.gov/9390366/" target="_blank" rel="noreferrer">Ishihara 선별 효율 연구</a>
        {" · "}
        <a className="underline underline-offset-2" href="https://www.nei.nih.gov/eye-health-information/eye-conditions-and-diseases/color-blindness/testing-color-vision-deficiency" target="_blank" rel="noreferrer">미국 NEI 안내</a>
      </div>
    </div>
  );
}
