"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useSession } from "next-auth/react";
import { useModel } from "../context/ModelContext";
import { CVDType } from "../hooks/useCVDModel";
import {
  Plate,
  plateSrc,
  pickPlates,
  diagnose,
  diagnosisBars,
  DIAGNOSIS_META,
  Bar,
} from "@/lib/ishiharaPlates";

// AI 보정 미리보기 카드 — 결과 화면에서 "틀린 도판"을 보정 상태로 다시 보여줘
// 보정의 가치를 데모한다. 진단 판정에는 절대 관여하지 않는다(판정은 보정 OFF 고정).
function PlateCorrectionCard({ plate, correctionType }: { plate: Plate; correctionType: CVDType }) {
  const { ready, infer } = useModel();
  const [on, setOn] = useState(false);
  const [correctedSrc, setCorrectedSrc] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  const apply = useCallback(async () => {
    if (!ready || !imgRef.current) return;
    setBusy(true);
    try {
      const SIZE = 300;
      const canvas = document.createElement("canvas");
      canvas.width = SIZE;
      canvas.height = SIZE;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(imgRef.current, 0, 0, SIZE, SIZE);
      const result = await infer(ctx.getImageData(0, 0, SIZE, SIZE), correctionType);
      const out = document.createElement("canvas");
      out.width = SIZE;
      out.height = SIZE;
      out.getContext("2d")!.putImageData(result, 0, 0);
      setCorrectedSrc(out.toDataURL());
    } catch (e) {
      console.error("보정 실패:", e);
      setOn(false);
    } finally {
      setBusy(false);
    }
  }, [ready, infer, correctionType]);

  useEffect(() => {
    if (on && !correctedSrc) apply();
  }, [on, correctedSrc, apply]);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-[150px] h-[150px]">
        {!on || !correctedSrc ? (
          <img
            ref={imgRef}
            src={plateSrc(plate)}
            alt={`이시하라 플레이트 ${plate.id}`}
            width={150}
            height={150}
            className="rounded-full border object-cover"
            style={{ borderColor: on ? "var(--color-brand)" : "var(--border)", width: 150, height: 150 }}
            crossOrigin="anonymous"
          />
        ) : (
          <img
            src={correctedSrc}
            alt="AI 보정"
            width={150}
            height={150}
            className="rounded-full border object-cover"
            style={{ borderColor: "var(--color-brand)", width: 150, height: 150 }}
          />
        )}
        {busy && (
          <div className="absolute inset-0 rounded-full bg-black/60 flex items-center justify-center">
            <p className="text-white text-xs">보정 중...</p>
          </div>
        )}
      </div>
      <p className="font-mono text-[11px]" style={{ color: "var(--fg-subtle)" }}>
        정답 <span style={{ color: "var(--fg)" }}>{plate.answer}</span>
      </p>
      <button
        onClick={() => setOn(!on)}
        disabled={!ready || busy}
        className="px-3 py-1 rounded-full text-[12px] font-medium transition-colors disabled:opacity-40"
        style={on
          ? { background: "var(--color-brand)", color: "#fff" }
          : { background: "var(--bg-muted)", color: "var(--fg)" }
        }
      >
        AI 보정 {on ? "ON" : "OFF"}
      </button>
    </div>
  );
}

// 유형별 색각이상 신호 막대그래프 — 경량 CSS 바(새 라이브러리 없음).
function DiagnosisBars({ bars }: { bars: Bar[] }) {
  return (
    <div className="w-full rounded-xl border p-5 flex flex-col gap-3" style={{ borderColor: "var(--border)", background: "var(--bg-elevated)" }}>
      <p className="text-sm font-medium" style={{ color: "var(--fg)" }}>분류판 응답 분포</p>
      <div className="flex flex-col gap-3 mt-1">
        {bars.map((b) => {
          const pct = b.max > 0 ? (b.value / b.max) * 100 : 0;
          const disabled = b.note != null;
          return (
            <div key={b.key} className="flex items-center gap-3">
              <span className="w-16 shrink-0 text-[13px]" style={{ color: "var(--fg-muted)" }}>{b.label}</span>
              <div className="flex-1 rounded-full h-4 overflow-hidden" style={{ background: "var(--bg-muted)" }} role="img"
                   aria-label={`${b.label} 신호 ${b.value} / ${b.max}`}>
                <div className="h-4 rounded-full transition-all" style={{ width: `${pct}%`, background: b.color, opacity: disabled ? 0.3 : 1 }} />
              </div>
              <span className="w-20 shrink-0 text-right font-mono text-[12px]" style={{ color: "var(--fg-subtle)" }}>
                {disabled ? b.note : `${b.value}/${b.max}`}
              </span>
            </div>
          );
        })}
      </div>
      <p className="text-[11px] font-mono mt-1" style={{ color: "var(--fg-subtle)" }}>
        ※ 분류판(정상·적색맹·녹색맹이 다른 숫자를 읽는 판)에서 각 유형의 답을 읽은 수입니다.
        정상 응답이 높으면 정상, 적색맹(빨강)·녹색맹(초록) 막대가 높을수록 해당 유형 가능성을 시사합니다.
      </p>
    </div>
  );
}

export default function IshiharaTest() {
  const { data: session } = useSession();

  const [plates, setPlates] = useState(pickPlates);
  const [current, setCurrent] = useState(0);
  const [answers, setAnswers] = useState<(string | null)[]>(() => Array(plates.length).fill(null));
  const [input, setInput] = useState("");
  const [done, setDone] = useState(false);

  const submittingRef = useRef(false);
  const plate = plates[current];

  useEffect(() => {
    setInput("");
  }, [current]);

  const submit = async () => {
    if (submittingRef.current) return;
    submittingRef.current = true;
    const newAnswers = [...answers];
    newAnswers[current] = input.trim() || "모름";
    setAnswers(newAnswers);

    if (current + 1 >= plates.length) {
      const result = diagnose(plates, newAnswers);
      const correct = newAnswers.filter((ans, i) => ans === plates[i].answer).length;

      if (session?.user) {
        await fetch("/api/results", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ correct, total: plates.length, diagnosis: result.type }),
        }).catch(() => {});
      }
      setDone(true);
    } else {
      setCurrent(current + 1);
    }
    submittingRef.current = false;
  };

  const restart = () => {
    const next = pickPlates();
    setPlates(next);
    setAnswers(Array(next.length).fill(null));
    setCurrent(0);
    setDone(false);
  };

  if (done) {
    const result = diagnose(plates, answers);
    const meta = DIAGNOSIS_META[result.type];
    const bars = diagnosisBars(result);
    const correct = answers.filter((a, i) => a === plates[i].answer).length;
    const isDeficient = result.type !== "normal";
    const correctionType = (meta.correctionType ?? "d") as CVDType;
    const wrongPlates = plates.filter((p, i) => p.kind !== "demo" && answers[i] !== p.answer);

    return (
      <div className="flex flex-col items-center gap-6 py-6 w-full">
        <div className="text-center">
          <p className="font-serif text-[26px] tracking-[-0.02em] mb-2">검사 완료</p>
          <p
            className="text-[15px] font-medium px-4 py-1.5 rounded-full inline-block"
            style={{ background: meta.color + "18", color: meta.color }}
          >
            {meta.label}
          </p>
          <p className="text-xs mt-3 font-mono" style={{ color: "var(--fg-subtle)" }}>
            ※ 적록 계열 선별·감별 결과입니다. 청색 계열(청색맹)은 감별 범위 밖이며,
            정확한 진단은 안과 전문의에게 받으세요.
          </p>
        </div>

        {/* 유형별 신호 막대그래프 */}
        <DiagnosisBars bars={bars} />

        <div className="w-full rounded-xl overflow-hidden border" style={{ borderColor: "var(--border)" }}>
          <div
            className="grid grid-cols-[2rem_3rem_3rem_1fr] gap-3 px-4 py-2.5 font-mono text-[11px] tracking-wide border-b"
            style={{ background: "var(--bg-muted)", color: "var(--fg-subtle)", borderColor: "var(--border)" }}
          >
            <span>#</span>
            <span>정답</span>
            <span>내 답</span>
            <span>결과</span>
          </div>
          {plates.map((p, i) => {
            const isCorrect = answers[i] === p.answer;
            return (
              <div
                key={p.id}
                className="grid grid-cols-[2rem_3rem_3rem_1fr] gap-3 px-4 py-3 border-b last:border-0 text-sm items-center"
                style={{ borderColor: "var(--border)" }}
              >
                <span className="font-mono text-[12px]" style={{ color: "var(--fg-subtle)" }}>{i + 1}</span>
                <div className="flex flex-col">
                  <span className="font-mono font-medium">{p.answer}</span>
                  {p.kind === "classification" && (
                    <span className="font-mono text-[10px]" style={{ color: "var(--fg-subtle)" }}>
                      적{p.protan}·녹{p.deutan}
                    </span>
                  )}
                </div>
                <span className="font-mono font-medium" style={{ color: isCorrect ? "#22c55e" : "#ef4444" }}>
                  {answers[i] ?? "-"}
                </span>
                <div>
                  {isCorrect ? (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono" style={{ background: "#22c55e18", color: "#22c55e" }}>
                      ✓ 정상
                    </span>
                  ) : p.kind === "demo" ? (
                    <span className="text-xs font-mono" style={{ color: "var(--fg-subtle)" }}>-</span>
                  ) : (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono" style={{ background: "#f9731618", color: "#f97316" }}>
                      ✗ 오답
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div className="w-full rounded-xl p-4" style={{ background: "var(--bg-muted)" }}>
          <p className="text-sm font-mono mb-1" style={{ color: "var(--fg-subtle)" }}>
            정답 {correct} / {plates.length} · 적록 선별 오답 {result.screeningErrors}/{result.screeningTotal}
            {result.classTotal > 0 && ` · 분류판 적${result.protanVotes}·녹${result.deutanVotes}`}
          </p>
          {isDeficient && (
            <p className="text-sm" style={{ color: "var(--fg-muted)" }}>
              적록 선별·분류에서 색각이상 신호가 나타납니다. 안과 정밀 검사를 권장합니다.
            </p>
          )}
        </div>

        {/* AI 보정으로 다시 보기 — 틀린 도판을 보정 상태로 재표시(진단 판정과 무관) */}
        {wrongPlates.length > 0 && (
          <div className="w-full rounded-xl border p-5 flex flex-col gap-4" style={{ borderColor: "var(--border)", background: "var(--bg-elevated)" }}>
            <div>
              <p className="text-sm font-medium mb-1" style={{ color: "var(--fg)" }}>AI 보정으로 다시 보기</p>
              <p className="text-[13px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>
                읽기 어려웠던 도판을 AI 보정 상태로 확인해보세요. 보정이 색 대비를 어떻게
                되살리는지 볼 수 있습니다. (이 미리보기는 진단 결과에 반영되지 않습니다.)
              </p>
            </div>
            <div className="flex flex-wrap gap-6 justify-center">
              {wrongPlates.map((p) => (
                <PlateCorrectionCard key={p.id} plate={p} correctionType={correctionType} />
              ))}
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-center justify-center gap-3">
          {isDeficient && (
            <Link
              href={`/correction?tab=image&type=${correctionType}`}
              className="px-6 py-2.5 rounded-full text-sm font-medium text-white inline-flex items-center gap-2"
              style={{ background: "var(--color-brand)" }}
            >
              이 유형으로 이미지 보정 해보기
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h14" />
                <path d="m13 6 6 6-6 6" />
              </svg>
            </Link>
          )}
          <Link
            href="/education"
            className="px-6 py-2.5 rounded-full text-sm font-medium inline-flex items-center gap-2"
            style={{ background: "var(--bg-muted)", color: "var(--fg)", border: "1px solid var(--border-strong)" }}
          >
            색각이상 알아보기
          </Link>
          <button
            onClick={restart}
            className="px-6 py-2.5 rounded-full text-sm font-medium transition-colors"
            style={isDeficient
              ? { background: "var(--bg-muted)", color: "var(--fg)", border: "1px solid var(--border-strong)" }
              : { background: "var(--color-brand)", color: "#fff" }
            }
          >
            다시 검사 (새 문제)
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-6">
      {/* 진행 상태 */}
      <div className="w-full flex items-center gap-3">
        <div className="flex-1 rounded-full h-1.5" style={{ background: "var(--border)" }}>
          <div
            className="h-1.5 rounded-full transition-all"
            style={{ width: `${(current / plates.length) * 100}%`, background: "var(--color-brand)" }}
          />
        </div>
        <span className="text-sm font-mono" style={{ color: "var(--fg-subtle)" }}>{current + 1} / {plates.length}</span>
      </div>

      <p className="text-sm" style={{ color: "var(--fg-muted)" }}>{plate.label}</p>

      {/* 플레이트 이미지 — 진단 단계에서는 항상 원본(보정 OFF) */}
      <div className="relative w-[280px] h-[280px]">
        <img
          src={plateSrc(plate)}
          alt={`이시하라 플레이트 ${plate.id}`}
          width={280}
          height={280}
          className="rounded-full border object-cover"
          style={{ borderColor: "var(--border)", width: 280, height: 280 }}
        />
      </div>

      {/* 답 입력 */}
      <div className="flex flex-col items-center gap-3 w-full max-w-xs">
        <p className="text-sm" style={{ color: "var(--fg-muted)" }}>보이는 숫자를 입력하세요</p>
        <div className="flex gap-2 w-full">
          <input
            type="text"
            inputMode="numeric"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
            placeholder="숫자 입력"
            className="flex-1 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand/30 border"
            style={{ background: "var(--bg-muted)", borderColor: "var(--border-strong)", color: "var(--fg)" }}
          />
          <button
            onClick={submit}
            className="px-4 py-2.5 rounded-lg text-sm font-medium text-white"
            style={{ background: "var(--color-brand)" }}
          >
            {current + 1 < plates.length ? "다음" : "완료"}
          </button>
        </div>
        <button
          onClick={() => { setInput("모름"); setTimeout(submit, 0); }}
          className="text-xs transition-colors"
          style={{ color: "var(--fg-subtle)" }}
          onMouseEnter={(e) => (e.currentTarget.style.color = "var(--fg)")}
          onMouseLeave={(e) => (e.currentTarget.style.color = "var(--fg-subtle)")}
        >
          잘 모르겠어요 (건너뛰기)
        </button>
      </div>
    </div>
  );
}
