"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { simulate, CVDType } from "@/lib/cvdSim";

type SimKey = "normal" | CVDType;

const SIM_OPTIONS: { key: SimKey; label: string; hint: string }[] = [
  { key: "normal", label: "정상 색각", hint: "일반적인 3원추 색각" },
  { key: "p", label: "적색맹 (P)", hint: "L-원추 결손 · 적록 혼동" },
  { key: "d", label: "녹색맹 (D)", hint: "M-원추 결손 · 적록 혼동" },
  { key: "t", label: "청색맹 (T)", hint: "S-원추 결손 · 청황 혼동" },
];

const TYPE_CARDS = [
  { code: "P", name: "적색맹 (Protanopia)", cone: "L-원추 (장파장·빨강)", axis: "적록", desc: "빨강 계열의 밝기·채도가 떨어져 빨강과 초록, 갈색과 녹색을 혼동합니다." },
  { code: "D", name: "녹색맹 (Deuteranopia)", cone: "M-원추 (중파장·초록)", axis: "적록", desc: "가장 흔한 유형으로, 적록 혼동 양상이 적색맹과 매우 비슷합니다." },
  { code: "T", name: "청색맹 (Tritanopia)", cone: "S-원추 (단파장·파랑)", axis: "청황", desc: "매우 드문 유형으로, 파랑과 초록·노랑과 분홍을 혼동합니다. 적록과는 완전히 다른 축입니다." },
];

// 유형 선택 → 동일 샘플 이미지의 CVD 시뮬레이션 (클라이언트 cvdSim, API 호출 없음).
function TypeSimExplorer() {
  const [sel, setSel] = useState<SimKey>("d");
  const [ready, setReady] = useState(false);
  const baseRef = useRef<ImageData | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // 기존 랜딩 에셋을 한 번만 로드해 ImageData로 캐시(새 이미지 추가 없음).
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      const W = Math.min(720, img.naturalWidth);
      const H = Math.round((img.naturalHeight / img.naturalWidth) * W);
      const c = document.createElement("canvas");
      c.width = W; c.height = H;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(img, 0, 0, W, H);
      baseRef.current = ctx.getImageData(0, 0, W, H);
      if (canvasRef.current) { canvasRef.current.width = W; canvasRef.current.height = H; }
      setReady(true);
    };
    img.src = "/landing/hero_original.jpg";
  }, []);

  // 선택 유형에 맞춰 캔버스 갱신.
  useEffect(() => {
    if (!ready || !baseRef.current || !canvasRef.current) return;
    const base = baseRef.current;
    const out = sel === "normal" ? base : simulate(base, sel);
    canvasRef.current.getContext("2d")!.putImageData(out, 0, 0);
  }, [sel, ready]);

  return (
    <div className="rounded-[20px] border p-5 md:p-7 flex flex-col gap-5" style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}>
      <div className="flex flex-wrap gap-2">
        {SIM_OPTIONS.map((o) => (
          <button
            key={o.key}
            onClick={() => setSel(o.key)}
            className="px-4 py-2 rounded-full text-[13px] font-medium transition-colors border"
            style={sel === o.key
              ? { background: "var(--color-brand)", color: "#fff", borderColor: "var(--color-brand)" }
              : { background: "transparent", color: "var(--fg-muted)", borderColor: "var(--border-strong)" }}
            aria-pressed={sel === o.key}
          >
            {o.label}
          </button>
        ))}
      </div>

      <div className="grid sm:grid-cols-2 gap-4">
        <figure className="flex flex-col gap-2">
          <img src="/landing/hero_original.jpg" alt="원본 이미지" className="w-full rounded-xl border" style={{ borderColor: "var(--border)" }} />
          <figcaption className="font-mono text-[11px]" style={{ color: "var(--fg-subtle)" }}>원본 (정상 색각)</figcaption>
        </figure>
        <figure className="flex flex-col gap-2">
          <div className="relative w-full rounded-xl border overflow-hidden" style={{ borderColor: "var(--border)", background: "var(--bg-muted)" }}>
            <canvas ref={canvasRef} className="w-full block" />
            {!ready && <div className="absolute inset-0 flex items-center justify-center text-xs" style={{ color: "var(--fg-subtle)" }}>불러오는 중…</div>}
          </div>
          <figcaption className="font-mono text-[11px]" style={{ color: "var(--fg-subtle)" }}>
            {SIM_OPTIONS.find((o) => o.key === sel)!.label} · {SIM_OPTIONS.find((o) => o.key === sel)!.hint}
          </figcaption>
        </figure>
      </div>
      <p className="text-[12px] font-mono" style={{ color: "var(--fg-subtle)" }}>
        ※ Brettel(1997) 모델 기반 클라이언트 시뮬레이션입니다. 정상 색각자가 각 유형의 시야를 근사적으로 체험하기 위한 것으로, 실제 지각과 다를 수 있습니다.
      </p>
    </div>
  );
}

function Section({ eyebrow, title, children }: { eyebrow: string; title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-4">
      <div>
        <p className="font-mono text-[11px] tracking-[0.16em] uppercase mb-2" style={{ color: "var(--fg-subtle)" }}>{eyebrow}</p>
        <h2 className="font-serif text-[28px] md:text-[36px] leading-[1.1] tracking-[-0.02em]">{title}</h2>
      </div>
      {children}
    </section>
  );
}

export default function EducationPage() {
  return (
    <div className="max-w-[1080px] mx-auto px-6 md:px-10 py-10 md:py-14 pb-32">
      {/* Breadcrumbs */}
      <div className="flex items-center gap-2 mb-8 font-mono text-[11px] tracking-[0.08em]" style={{ color: "var(--fg-subtle)" }}>
        <Link href="/" className="hover:text-fg transition-colors">HOME</Link>
        <span>/</span>
        <span style={{ color: "var(--fg)" }}>02 · LEARN</span>
      </div>

      {/* Hero */}
      <div className="mb-14">
        <p className="font-mono text-[11px] tracking-[0.16em] uppercase mb-3" style={{ color: "var(--fg-subtle)" }}>GUIDE TO COLOR VISION</p>
        <h1 className="font-serif text-[40px] md:text-[64px] leading-[1.03] tracking-[-0.03em] mb-5">색각이상<br />이해하기</h1>
        <p className="text-[15px] md:text-[17px] leading-relaxed max-w-[640px]" style={{ color: "var(--fg-muted)" }}>
          색각이상은 특정 색을 &lsquo;못 보는&rsquo; 것이 아니라, 색으로 구분되던 정보의 <strong style={{ color: "var(--fg)" }}>대비(contrast)</strong>를
          잃는 상태입니다. 원인과 유형, 일상에서의 영향, 그리고 우리 서비스가 어떻게 대비를 되살리는지 알아봅니다.
        </p>
      </div>

      <div className="flex flex-col gap-16">
        {/* 원인 */}
        <Section eyebrow="Mechanism" title="원인 — 원추세포의 이상">
          <p className="text-[15px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>
            사람의 망막에는 파장대가 다른 세 종류의 원추세포(cone) — <strong style={{ color: "var(--fg)" }}>L(빨강)·M(초록)·S(파랑)</strong> — 가 있고,
            뇌는 이 세 신호의 비율로 색을 구분합니다. 특정 원추세포가 없거나 분광 민감도가 어긋나면 특정 색 쌍의 신호 차이가 줄어들어
            그 색들을 혼동하게 됩니다. 대부분 X염색체 유전에 따른 선천성이라, 남성에게 더 흔합니다.
          </p>
          <div className="grid sm:grid-cols-3 gap-3">
            {[["L-원추", "장파장 · 빨강"], ["M-원추", "중파장 · 초록"], ["S-원추", "단파장 · 파랑"]].map(([a, b]) => (
              <div key={a} className="rounded-xl border p-4" style={{ borderColor: "var(--border)", background: "var(--bg-elevated)" }}>
                <p className="font-mono text-[13px] font-medium" style={{ color: "var(--fg)" }}>{a}</p>
                <p className="text-[13px] mt-1" style={{ color: "var(--fg-muted)" }}>{b}</p>
              </div>
            ))}
          </div>
        </Section>

        {/* 유형 + 인터랙티브 */}
        <Section eyebrow="Types" title="유형 — 무엇을 혼동하는가">
          <div className="grid md:grid-cols-3 gap-3">
            {TYPE_CARDS.map((t) => (
              <div key={t.code} className="rounded-xl border p-5 flex flex-col gap-2" style={{ borderColor: "var(--border)", background: "var(--bg-elevated)" }}>
                <div className="flex items-center gap-2">
                  <span className="w-8 h-8 rounded-full flex items-center justify-center font-mono text-[13px] font-semibold text-white" style={{ background: t.axis === "적록" ? "#f97316" : "#2a5fd9" }}>{t.code}</span>
                  <p className="text-[14px] font-medium" style={{ color: "var(--fg)" }}>{t.name}</p>
                </div>
                <p className="font-mono text-[11px]" style={{ color: "var(--fg-subtle)" }}>{t.cone} · {t.axis} 혼동축</p>
                <p className="text-[13px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>{t.desc}</p>
              </div>
            ))}
          </div>

          {/* 적록 두 유형이 유사한 이유 (pd_similarity 일반인용) */}
          <div className="rounded-xl p-5 border-l-4" style={{ background: "var(--bg-muted)", borderColor: "#f97316" }}>
            <p className="text-[14px] font-medium mb-1" style={{ color: "var(--fg)" }}>적색맹(P)과 녹색맹(D)이 비슷해 보이는 이유</p>
            <p className="text-[13px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>
              둘 다 <strong style={{ color: "var(--fg)" }}>적록(빨강–초록) 계열</strong> 결손이라 &lsquo;혼동하는 색의 방향&rsquo;이 거의 같습니다.
              그래서 보이는 세계도, 우리의 보정 방향도 P와 D가 서로 닮습니다 — 버그가 아니라 색채학적으로 자연스러운 결과입니다.
              반면 청색맹(T)은 혼동축이 완전히 달라 확연히 구분됩니다.
            </p>
          </div>

          <div className="mt-2 flex flex-col gap-3">
            <p className="text-[14px] font-medium" style={{ color: "var(--fg)" }}>직접 체험 — 유형을 바꿔보세요</p>
            <TypeSimExplorer />
          </div>
        </Section>

        {/* 유병률 */}
        <Section eyebrow="Prevalence" title="얼마나 흔한가">
          <div className="grid sm:grid-cols-3 gap-3">
            {[["약 8%", "적록 계열 (남성)"], ["약 0.5%", "적록 계열 (여성)"], ["매우 드묾", "청색 계열 (T)"]].map(([a, b]) => (
              <div key={b} className="rounded-xl border p-5" style={{ borderColor: "var(--border)", background: "var(--bg-elevated)" }}>
                <p className="font-serif text-[30px] leading-none" style={{ color: "var(--color-brand)" }}>{a}</p>
                <p className="text-[13px] mt-2" style={{ color: "var(--fg-muted)" }}>{b}</p>
              </div>
            ))}
          </div>
          <p className="text-[12px] font-mono" style={{ color: "var(--fg-subtle)" }}>※ 대략적인 통계이며 인구·집단에 따라 차이가 있습니다.</p>
        </Section>

        {/* 일상 영향 */}
        <Section eyebrow="Impact" title="일상에서의 영향">
          <div className="grid sm:grid-cols-2 gap-3">
            {[
              ["신호·표지", "빨강·초록 신호등, 경고색 구분이 어려울 수 있습니다."],
              ["데이터·지도", "색으로만 구분한 그래프 범례·지하철 노선도가 뭉개져 보입니다."],
              ["음식", "덜 익은 과일과 익은 과일, 고기의 익힘 정도 판단이 어렵습니다."],
              ["학습·업무", "색 코딩된 자료, 색연필·형광펜 구분에서 불편을 겪습니다."],
            ].map(([a, b]) => (
              <div key={a} className="rounded-xl border p-5" style={{ borderColor: "var(--border)", background: "var(--bg-elevated)" }}>
                <p className="text-[14px] font-medium mb-1" style={{ color: "var(--fg)" }}>{a}</p>
                <p className="text-[13px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>{b}</p>
              </div>
            ))}
          </div>
        </Section>

        {/* 실생활 팁 */}
        <Section eyebrow="Tips" title="실생활 팁">
          <ul className="flex flex-col gap-2">
            {[
              "색에만 의존하지 말고 라벨·모양·위치 등 이중 단서를 함께 활용하세요.",
              "고대비 테마·다크모드, 색각 보조 필터를 적극 사용하세요.",
              "중요한 색 판단(신호·의약품 등)은 주변에 확인을 요청하는 습관이 도움이 됩니다.",
              "CVDLens 같은 실시간 보정 도구로 사진·화면의 색 대비를 되살릴 수 있습니다.",
            ].map((t) => (
              <li key={t} className="flex items-start gap-3 rounded-xl p-4" style={{ background: "var(--bg-muted)" }}>
                <span className="mt-0.5 shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-white text-[11px]" style={{ background: "var(--color-brand)" }}>✓</span>
                <span className="text-[14px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>{t}</span>
              </li>
            ))}
          </ul>
        </Section>

        {/* CTA */}
        <div className="rounded-[20px] border p-7 md:p-9 flex flex-col sm:flex-row sm:items-center gap-5 justify-between" style={{ borderColor: "var(--border)", background: "var(--bg-elevated)" }}>
          <div>
            <p className="font-serif text-[24px] tracking-[-0.02em] mb-1">직접 확인해 보세요</p>
            <p className="text-[14px]" style={{ color: "var(--fg-muted)" }}>자가 진단으로 유형을 가늠하고, AI 보정으로 대비를 되살려 보세요.</p>
          </div>
          <div className="flex flex-wrap gap-3 shrink-0">
            <Link href="/ishihara" className="px-6 py-2.5 rounded-full text-sm font-medium text-white" style={{ background: "var(--color-brand)" }}>색각 진단하기</Link>
            <Link href="/correction?tab=image" className="px-6 py-2.5 rounded-full text-sm font-medium" style={{ background: "var(--bg-muted)", color: "var(--fg)", border: "1px solid var(--border-strong)" }}>이미지 보정 체험</Link>
          </div>
        </div>
      </div>

      <p className="mt-12 text-xs font-mono tracking-[0.04em]" style={{ color: "var(--fg-subtle)" }}>
        ※ 본 페이지는 교육·참고용입니다. 정확한 진단은 안과 전문의에게 문의하세요.
      </p>
    </div>
  );
}
