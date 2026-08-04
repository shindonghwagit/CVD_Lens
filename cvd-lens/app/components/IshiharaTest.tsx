"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useSession } from "next-auth/react";
import { useModel } from "../context/ModelContext";
import { CVDType } from "../hooks/useCVDModel";

// answer: 정상 색각자가 읽는 숫자
// altAnswer: 색각이상자가 읽는 숫자 (null=읽지 못함)
// type: demo=전원정답 / pd=적록(적색맹·녹색맹) 스크리닝 / t=청-노 계열
//   표준 이시하라 도판은 적록 스크리닝용이며 청색(tritan) 감별력이 검증되지
//   않았다. 그래서 'pd'만 실제 진단 풀로 쓰고, 't'로 표시된 도판(11)은 데이터로
//   보존하되 스크리닝/판정에서 제외한다(검증된 tritan 도판 확보는 별도 과제).
// ext: 파일 확장자 (기본 jpg)
const ALL_PLATES = [
  { id: "01", answer: "74",  altAnswer: "21",  type: "demo" as const, label: "모든 분이 읽을 수 있는 플레이트" },
  { id: "03", answer: "16",  altAnswer: null,   type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "04", answer: "2",   altAnswer: null,   type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "05", answer: "29",  altAnswer: null,   type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "06", answer: "7",   altAnswer: null,   type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "07", answer: "45",  altAnswer: null,   type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "08", answer: "5",   altAnswer: "2",    type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "09", answer: "97",  altAnswer: null,   type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "10", answer: "8",   altAnswer: null,   type: "pd"   as const, label: "적록색각이상 감별" },
  // 11: 청-노 계열로 분류돼 있으나 표준 이시하라의 tritan 감별력 미검증 → 진단 풀 제외.
  { id: "11", answer: "42",  altAnswer: null,   type: "t"    as const, label: "청색 계열 (참고용, 진단 제외)" },
  { id: "12", answer: "3",   altAnswer: null,   type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "13", answer: "42",  altAnswer: null,   type: "pd"   as const, label: "적록색각이상 감별", ext: "png" },
  { id: "14", answer: "27",  altAnswer: null,   type: "pd"   as const, label: "적록색각이상 감별", ext: "png" },
  { id: "15", answer: "12",  altAnswer: null,   type: "pd"   as const, label: "적록색각이상 감별", ext: "png" },
];

type Plate = (typeof ALL_PLATES)[number];

const plateSrc = (p: Plate) => `/ishihara/Ishihara_${p.id}.${"ext" in p && p.ext ? p.ext : "jpg"}`;

function pickPlates() {
  const demo = ALL_PLATES.filter((p) => p.type === "demo");
  // 적록(pd) 도판만 진단 풀로 사용 — 표준 이시하라의 검증된 축.
  const diagnostic = ALL_PLATES.filter((p) => p.type === "pd");
  const shuffled = [...diagnostic].sort(() => Math.random() - 0.5);
  return [...demo, ...shuffled.slice(0, 4)]; // 1 demo + 4 랜덤 = 5장
}

// AI 보정 미리보기 카드 — 결과 화면에서 "틀린 도판"을 보정 상태로 다시 보여줘
// 보정의 가치를 데모한다. 진단 판정에는 절대 관여하지 않는다(판정은 보정 OFF 고정).
function PlateCorrectionCard({ plate }: { plate: Plate }) {
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
      // 적록 스크리닝이므로 녹색맹(d) 보정으로 데모.
      const result = await infer(ctx.getImageData(0, 0, SIZE, SIZE), "d" as CVDType);
      const out = document.createElement("canvas");
      out.width = SIZE;
      out.height = SIZE;
      out.getContext("2d")!.putImageData(result, 0, 0);
      setCorrectedSrc(out.toDataURL());
    } catch (e) {
      // 서버 미가용 등 실패 시 원본 유지 + 토글 원복(스피너 고착 방지).
      console.error("보정 실패:", e);
      setOn(false);
    } finally {
      setBusy(false);
    }
  }, [ready, infer]);

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

const TYPE_LABEL: Record<string, { short: string; color: string }> = {
  pd: { short: "적록색각이상 의심", color: "#f97316" },
};

export default function IshiharaTest() {
  const { data: session } = useSession();

  const [plates, setPlates] = useState(pickPlates);
  const [current, setCurrent] = useState(0);
  const [answers, setAnswers] = useState<(string | null)[]>(() => Array(5).fill(null));
  const [input, setInput] = useState("");
  const [done, setDone] = useState(false);

  const submittingRef = useRef(false);
  const plate = plates[current];

  useEffect(() => {
    setInput("");
  }, [current]);

  // 판정은 적록(pd) 도판만 사용. 진단 단계는 보정 OFF 고정이라 보정으로 읽은 답이
  // 판정에 섞이지 않는다.
  function getDiagnosis(ans: (string | null)[]) {
    let pdWrong = 0;
    ans.forEach((a, i) => {
      if (plates[i].type !== "pd") return;
      if (a !== plates[i].answer) pdWrong++;
    });
    const pdCount = plates.filter((p) => p.type === "pd").length;
    const threshold = Math.max(1, Math.round(pdCount / 2));
    return pdWrong >= threshold ? "d" : "normal";
  }

  const submit = async () => {
    if (submittingRef.current) return;
    submittingRef.current = true;
    const newAnswers = [...answers];
    newAnswers[current] = input.trim() || "모름";
    setAnswers(newAnswers);

    if (current + 1 >= plates.length) {
      const correct = newAnswers.filter((ans, i) => ans === plates[i].answer).length;
      const diagnosis = getDiagnosis(newAnswers);

      if (session?.user) {
        await fetch("/api/results", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ correct, total: plates.length, diagnosis }),
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
    const diagnosis = getDiagnosis(answers);
    const correct = answers.filter((a, i) => a === plates[i].answer).length;
    const diagnosisColor = diagnosis === "d" ? "#f97316" : "#22c55e";
    const diagnosisLabel =
      diagnosis === "d"
        ? "적록색각이상 (적색맹/녹색맹) 가능성이 있습니다."
        : "적록 계열 색각이상 징후는 보이지 않습니다.";
    const wrongPlates = plates.filter((p, i) => p.type !== "demo" && answers[i] !== p.answer);

    return (
      <div className="flex flex-col items-center gap-6 py-6 w-full">
        <div className="text-center">
          <p className="font-serif text-[26px] tracking-[-0.02em] mb-2">진단 완료</p>
          <p
            className="text-[15px] font-medium px-4 py-1.5 rounded-full inline-block"
            style={{ background: diagnosisColor + "18", color: diagnosisColor }}
          >
            {diagnosisLabel}
          </p>
          <p className="text-xs mt-3 font-mono" style={{ color: "var(--fg-subtle)" }}>
            ※ 적록 계열 스크리닝 결과입니다. 청색 계열(청색맹)은 감별 범위 밖이며,
            정확한 진단은 안과 전문의에게 받으세요.
          </p>
        </div>

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
            const typeInfo = p.type !== "demo" ? TYPE_LABEL[p.type] : null;
            return (
              <div
                key={p.id}
                className="grid grid-cols-[2rem_3rem_3rem_1fr] gap-3 px-4 py-3 border-b last:border-0 text-sm items-center"
                style={{ borderColor: "var(--border)" }}
              >
                <span className="font-mono text-[12px]" style={{ color: "var(--fg-subtle)" }}>{i + 1}</span>
                <div className="flex flex-col">
                  <span className="font-mono font-medium">{p.answer}</span>
                  {p.type !== "demo" && p.altAnswer && (
                    <span className="font-mono text-[10px]" style={{ color: "var(--fg-subtle)" }}>
                      색각이상: {p.altAnswer}
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
                  ) : p.type === "demo" || !typeInfo ? (
                    <span className="text-xs font-mono" style={{ color: "var(--fg-subtle)" }}>-</span>
                  ) : (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono" style={{ background: typeInfo.color + "18", color: typeInfo.color }}>
                      ✗ {typeInfo.short}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div className="w-full rounded-xl p-4" style={{ background: "var(--bg-muted)" }}>
          <p className="text-sm font-mono mb-1" style={{ color: "var(--fg-subtle)" }}>
            정답 {correct} / {plates.length}
          </p>
          {diagnosis === "d" && (
            <p className="text-sm" style={{ color: "var(--fg-muted)" }}>
              적록 도판에서 절반 이상 오답입니다. 안과 정밀 검사를 권장합니다.
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
                <PlateCorrectionCard key={p.id} plate={p} />
              ))}
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-center justify-center gap-3">
          {diagnosis === "d" && (
            <Link
              href="/correction?tab=image&type=d"
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
          <button
            onClick={restart}
            className="px-6 py-2.5 rounded-full text-sm font-medium transition-colors"
            style={diagnosis === "d"
              ? { background: "var(--bg-muted)", color: "var(--fg)", border: "1px solid var(--border-strong)" }
              : { background: "var(--color-brand)", color: "#fff" }
            }
          >
            다시 테스트 (새 문제)
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
