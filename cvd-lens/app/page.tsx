"use client";

import Link from "next/link";
import HeroVisual from "./components/HeroVisual";
import { useEffect, useRef } from "react";

function useFadeIn() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.style.opacity = "1";
          el.style.transform = "translateY(0)";
          observer.disconnect();
        }
      },
      { threshold: 0.12 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return ref;
}

const CVD_TYPES = [
  { type: "PROTAN", kor: "적색맹" },
  { type: "DEUTAN", kor: "녹색맹" },
  { type: "TRITAN", kor: "청색맹" },
];

const FEATURES = [
  {
    num: "01 / ASSESSMENT",
    href: "/ishihara",
    title: "이시하라 색각 진단",
    subtitle: "4장의 이시하라 도트판으로 색각이상 여부를 스스로 확인합니다.",
    detail: "제1·2형(적록) 및 제3형(청황) 감별 · 결과는 참고용이며 정확한 진단은 안과 전문의와 상담하세요.",
    tags: ["4 plates", "~2 min", "AI 보정 토글"],
  },
  {
    num: "02 / IMAGE",
    href: "/correction?tab=image",
    title: "이미지 파일 보정",
    subtitle: "사진을 업로드하면 AI가 색각이상 타입별로 보정된 결과를 생성합니다.",
    detail: "원본과 보정본을 슬라이더로 비교 · Protan / Deutan / Tritan 3종 지원 · FastAPI 서버 추론.",
    tags: ["jpg / png", "슬라이더 비교", "3종 타입"],
  },
  {
    num: "03 / CAMERA",
    href: "/correction",
    title: "카메라 촬영 보정",
    subtitle: "카메라로 사진을 찍으면 AI가 색각이상 타입에 맞춰 색 대비를 보정합니다.",
    detail: "MobileNetV2 U-Net · FastAPI 서버 추론 · 슬라이더로 원본과 보정본 비교.",
    tags: ["Camera", "사진 촬영", "AI 보정"],
  },
];

export default function Home() {
  const featuresRef = useFadeIn();

  return (
    <div className="max-w-[1280px] mx-auto px-6 md:px-10">
      {/* Hero */}
      <section className="grid md:grid-cols-[1.1fr_1fr] gap-12 md:gap-16 items-center py-16 md:py-20">
        <div>
          <p className="text-[11px] tracking-[0.16em] uppercase font-mono mb-5" style={{ color: "var(--fg-subtle)" }}>
            CVDLENS · COLOR VISION ASSIST
          </p>
          <h1 className="font-serif text-[44px] md:text-[72px] leading-[1.02] tracking-[-0.035em] mb-6">
            보이지 않던 <em className="not-italic text-brand italic" style={{ fontStyle: "italic" }}>색</em>을,
            <br />
            AI가 다시 <em className="not-italic text-brand italic" style={{ fontStyle: "italic" }}>또렷하게</em>.
          </h1>
          <p className="text-[17px] leading-relaxed mb-8 max-w-[480px]" style={{ color: "var(--fg-muted)" }}>
            색각이상(Color Vision Deficiency)을 가진 사용자를 위한 AI 보정 도구.
            자가 진단부터 사진·카메라 촬영 보정까지, 한 곳에서 해결하세요.
          </p>
          <div className="flex gap-3 mb-10">
            <Link href="/ishihara" className="inline-flex items-center gap-2 px-5 py-3 rounded-full bg-brand hover:bg-brand-ink text-white text-sm font-medium transition-colors">
              자가 진단 시작
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h14" />
                <path d="m13 6 6 6-6 6" />
              </svg>
            </Link>
            <a
              href="#features"
              className="inline-flex items-center gap-2 px-5 py-3 rounded-full border text-sm transition-colors"
              style={{ borderColor: "var(--border-strong)", color: "var(--fg)" }}
            >
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 5v14" />
                <path d="m6 13 6 6 6-6" />
              </svg>
              3가지 기능 보기
            </a>
          </div>

          <div className="grid grid-cols-3 gap-2.5 max-w-[420px]">
            {CVD_TYPES.map((c) => (
              <div key={c.type} className="p-3 rounded-xl border bg-elevated" style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}>
                <div className="font-mono text-xs" style={{ color: "var(--fg-muted)" }}>{c.type}</div>
                <div className="mt-2 h-3 rounded" style={{ background: "linear-gradient(90deg, #d5383a, #d89a2b, #4b8d3b, #2d69b8, #6a49a8)" }} />
                <div className="font-mono text-[10px] mt-1.5" style={{ color: "var(--fg-subtle)" }}>{c.kor}</div>
              </div>
            ))}
          </div>
        </div>

        <HeroVisual />
      </section>

      {/* Feature stack */}
      <section
        id="features"
        ref={featuresRef}
        className="py-12"
        style={{ opacity: 0, transform: "translateY(32px)", transition: "opacity 0.6s ease, transform 0.6s ease" }}
      >
        <div className="flex flex-col md:flex-row md:justify-between md:items-end gap-6 mb-10">
          <div>
            <p className="text-[11px] tracking-[0.16em] uppercase font-mono mb-2" style={{ color: "var(--fg-subtle)" }}>
              THREE TOOLS · 한 번에
            </p>
            <h2 className="font-serif text-[32px] md:text-[48px] leading-[1.1] tracking-[-0.02em]">
              진단하고, 보정하고,
              <br />선명하게 다시 본다.
            </h2>
          </div>
          <p className="text-sm max-w-[320px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>
            아래 세 기능은 동일한 AI 보정 엔진을 공유합니다. 어디서 시작해도 같은 품질의 색 보정을 경험할 수 있어요.
          </p>
        </div>

        {FEATURES.map((f, i) => (
          <Link
            key={f.href}
            href={f.href}
            className="group grid md:grid-cols-[80px_1fr_1.2fr] gap-6 md:gap-10 py-12 border-t transition-all duration-200 hover:bg-brand-soft/50 hover:px-6 hover:rounded-2xl hover:border-transparent"
            style={{ borderColor: "var(--border)", animationDelay: `${i * 0.1}s` }}
          >
            <div className="font-mono text-xs tracking-[0.12em] pt-1" style={{ color: "var(--fg-subtle)" }}>
              {f.num}
            </div>
            <div>
              <h3 className="font-serif text-[28px] md:text-[36px] leading-[1.1] tracking-[-0.02em] mb-3">{f.title}</h3>
              <p className="text-[15px] leading-relaxed mb-2 max-w-[420px]" style={{ color: "var(--fg-muted)" }}>
                {f.subtitle}
              </p>
              <p className="text-[13px] max-w-[420px] mb-5" style={{ color: "var(--fg-subtle)" }}>
                {f.detail}
              </p>
              <div className="flex gap-1.5 flex-wrap">
                {f.tags.map((t) => (
                  <span
                    key={t}
                    className="font-mono text-[10px] px-2 py-0.5 rounded-full border"
                    style={{ background: "var(--bg-elevated)", borderColor: "var(--border)", color: "var(--fg-muted)" }}
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
            <div
              className="aspect-[16/10] rounded-2xl border relative overflow-hidden shadow-soft group-hover:-translate-y-1 group-hover:shadow-lift transition-all"
              style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}
            >
              {/* Plate preview for first card, others get placeholder */}
              {i === 0 && (
                <div className="absolute inset-0 flex items-center justify-center gap-4">
                  {["06", "01", "08"].map((id, j) => (
                    <div
                      key={id}
                      className="w-[130px] h-[130px] rounded-full overflow-hidden shadow-lift"
                      style={{ transform: `translateY(${j === 1 ? "-10px" : "0"}) rotate(${(j - 1) * 4}deg)` }}
                    >
                      <img
                        src={`/ishihara/Ishihara_${id}.jpg`}
                        alt={`이시하라 플레이트 ${id}`}
                        className="w-full h-full object-cover"
                      />
                    </div>
                  ))}
                </div>
              )}
              {i !== 0 && (
                <div
                  className="absolute inset-0 flex items-center justify-center font-mono text-[11px] tracking-[0.1em]"
                  style={{
                    background: "repeating-linear-gradient(45deg, var(--bg-muted), var(--bg-muted) 10px, var(--bg-elevated) 10px, var(--bg-elevated) 20px)",
                    color: "var(--fg-subtle)",
                  }}
                >
                  {i === 1 ? "IMAGE PREVIEW" : "CAMERA PREVIEW"}
                </div>
              )}

              <div
                className="absolute top-4 right-4 w-9 h-9 rounded-full bg-brand text-white flex items-center justify-center transition-transform group-hover:-rotate-45 group-hover:scale-105 z-10"
              >
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M5 12h14" />
                  <path d="m13 6 6 6-6 6" />
                </svg>
              </div>
            </div>
          </Link>
        ))}
      </section>

    </div>
  );
}