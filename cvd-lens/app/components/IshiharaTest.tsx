"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useSession } from "next-auth/react";
import { useModel } from "../context/ModelContext";
import { CVDType } from "../hooks/useCVDModel";

// answer: 정상 색각자가 읽는 숫자
// type: demo=전원정답, pd=적록색각이상 감별, t=청색맹 감별
// ext: 파일 확장자 (기본 jpg)
const PLATES = [
  { id: "01", answer: "74",  type: "demo" as const, label: "모든 분이 읽을 수 있는 플레이트" },
  { id: "03", answer: "16",  type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "04", answer: "2",   type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "05", answer: "29",  type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "06", answer: "7",   type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "07", answer: "45",  type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "08", answer: "5",   type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "09", answer: "97",  type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "10", answer: "8",   type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "11", answer: "42",  type: "t"    as const, label: "청색맹 감별" },
  { id: "12", answer: "3",   type: "pd"   as const, label: "적록색각이상 감별" },
  { id: "13", answer: "42",  type: "pd"   as const, label: "적록색각이상 감별", ext: "png" },
  { id: "14", answer: "27",  type: "pd"   as const, label: "적록색각이상 감별", ext: "png" },
  { id: "15", answer: "12",  type: "pd"   as const, label: "적록색각이상 감별", ext: "png" },
];

const TYPE_LABEL: Record<string, { short: string; color: string }> = {
  pd: { short: "적록색각이상 의심", color: "#f97316" },
  t:  { short: "청색맹 의심",      color: "#3b82f6" },
};

export default function IshiharaTest() {
  const { ready, infer } = useModel();
  const { data: session } = useSession();

  const [current, setCurrent] = useState(0);
  const [answers, setAnswers] = useState<(string | null)[]>(Array(PLATES.length).fill(null));
  const [input, setInput] = useState("");
  const [correctionOn, setCorrectionOn] = useState(false);
  const [correctedSrc, setCorrectedSrc] = useState<string | null>(null);
  const [correcting, setCorrecting] = useState(false);
  const [done, setDone] = useState(false);

  const submittingRef = useRef(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const plate = PLATES[current];

  useEffect(() => {
    setCorrectionOn(false);
    setCorrectedSrc(null);
    setInput("");
  }, [current]);

  const applyCorrection = useCallback(async () => {
    if (!ready || !imgRef.current) return;
    setCorrecting(true);

    const img = imgRef.current;
    const SIZE = 300;
    const canvas = document.createElement("canvas");
    canvas.width = SIZE;
    canvas.height = SIZE;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, 0, 0, SIZE, SIZE);

    const cvdType: CVDType = plate.type === "t" ? "t" : "d";
    const imageData = ctx.getImageData(0, 0, SIZE, SIZE);
    const result = await infer(imageData, cvdType);

    const out = document.createElement("canvas");
    out.width = SIZE;
    out.height = SIZE;
    out.getContext("2d")!.putImageData(result, 0, 0);
    setCorrectedSrc(out.toDataURL());
    setCorrecting(false);
  }, [ready, infer, plate]);

  useEffect(() => {
    if (correctionOn && !correctedSrc) applyCorrection();
  }, [correctionOn, correctedSrc, applyCorrection]);

  const submit = async () => {
    if (submittingRef.current) return;
    submittingRef.current = true;
    const newAnswers = [...answers];
    newAnswers[current] = input.trim() || "모름";
    setAnswers(newAnswers);

    if (current + 1 >= PLATES.length) {
      const correct = newAnswers.filter((ans, i) => ans === PLATES[i].answer).length;
      const diagnosis = getDiagnosis(newAnswers);

      if (session?.user) {
        await fetch("/api/results", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ correct, total: PLATES.length, diagnosis }),
        }).catch(() => {});
      }
      setDone(true);
    } else {
      setCurrent(current + 1);
    }
    submittingRef.current = false;
  };

  function getDiagnosis(ans: (string | null)[]) {
    let pdWrong = 0, tWrong = 0;
    ans.forEach((a, i) => {
      if (PLATES[i].type === "demo") return;
      if (a !== PLATES[i].answer) {
        if (PLATES[i].type === "pd") pdWrong++;
        else tWrong++;
      }
    });
    if (pdWrong >= 4) return "d";
    if (tWrong >= 1)  return "t";
    return "normal";
  }

  if (done) {
    const diagnosis = getDiagnosis(answers);
    const correct = answers.filter((a, i) => a === PLATES[i].answer).length;
    const diagnosisColor = diagnosis === "d" ? "#f97316" : diagnosis === "t" ? "#3b82f6" : "#22c55e";
    const diagnosisLabel =
      diagnosis === "d" ? "적록색각이상 (적색맹/녹색맹) 가능성이 있습니다." :
      diagnosis === "t" ? "청색맹 가능성이 있습니다." :
      "정상 색각으로 판단됩니다.";

    return (
      <div className="flex flex-col items-center gap-6 py-6 w-full">
        {/* 진단 결과 헤더 */}
        <div className="text-center">
          <p className="font-serif text-[26px] tracking-[-0.02em] mb-2">진단 완료</p>
          <p
            className="text-[15px] font-medium px-4 py-1.5 rounded-full inline-block"
            style={{ background: diagnosisColor + "18", color: diagnosisColor }}
          >
            {diagnosisLabel}
          </p>
          <p className="text-xs mt-3 font-mono" style={{ color: "var(--fg-subtle)" }}>
            ※ 참고용이며 정확한 진단은 안과 전문의에게 받으세요.
          </p>
        </div>

        {/* 플레이트별 상세 결과 */}
        <div className="w-full rounded-xl overflow-hidden border" style={{ borderColor: "var(--border)" }}>
          {/* 헤더 */}
          <div
            className="grid grid-cols-[2rem_3rem_3rem_1fr] gap-3 px-4 py-2.5 font-mono text-[11px] tracking-wide border-b"
            style={{ background: "var(--bg-muted)", color: "var(--fg-subtle)", borderColor: "var(--border)" }}
          >
            <span>#</span>
            <span>정답</span>
            <span>내 답</span>
            <span>결과</span>
          </div>

          {PLATES.map((p, i) => {
            const isCorrect = answers[i] === p.answer;
            const typeInfo = p.type !== "demo" ? TYPE_LABEL[p.type] : null;
            return (
              <div
                key={i}
                className="grid grid-cols-[2rem_3rem_3rem_1fr] gap-3 px-4 py-3 border-b last:border-0 text-sm items-center"
                style={{ borderColor: "var(--border)" }}
              >
                <span className="font-mono text-[12px]" style={{ color: "var(--fg-subtle)" }}>{i + 1}</span>
                <span className="font-mono font-medium">{p.answer}</span>
                <span className="font-mono font-medium" style={{ color: isCorrect ? "#22c55e" : "#ef4444" }}>
                  {answers[i] ?? "-"}
                </span>
                <div>
                  {isCorrect ? (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono" style={{ background: "#22c55e18", color: "#22c55e" }}>
                      ✓ 정상
                    </span>
                  ) : p.type === "demo" ? (
                    <span className="text-xs font-mono" style={{ color: "var(--fg-subtle)" }}>-</span>
                  ) : (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono" style={{ background: typeInfo!.color + "18", color: typeInfo!.color }}>
                      ✗ {typeInfo!.short}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* 요약 */}
        <div className="w-full rounded-xl p-4" style={{ background: "var(--bg-muted)" }}>
          <p className="text-sm font-mono mb-2" style={{ color: "var(--fg-subtle)" }}>
            정답 {correct} / {PLATES.length}
          </p>
          {diagnosis !== "normal" && (
            <p className="text-sm" style={{ color: "var(--fg-muted)" }}>
              {diagnosis === "d"
                ? "적록색각이상 플레이트에서 4개 이상 오답입니다. 안과 정밀 검사를 권장합니다."
                : "청색맹 플레이트에서 오답입니다. 안과 정밀 검사를 권장합니다."}
            </p>
          )}
        </div>

        <button
          onClick={() => { setCurrent(0); setAnswers(Array(PLATES.length).fill(null)); setDone(false); }}
          className="px-6 py-2.5 rounded-full text-sm font-medium text-white"
          style={{ background: "var(--color-brand)" }}
        >
          다시 테스트
        </button>
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
            style={{ width: `${(current / PLATES.length) * 100}%`, background: "var(--color-brand)" }}
          />
        </div>
        <span className="text-sm font-mono" style={{ color: "var(--fg-subtle)" }}>{current + 1} / {PLATES.length}</span>
      </div>

      <p className="text-sm" style={{ color: "var(--fg-muted)" }}>{plate.label}</p>

      {/* 플레이트 이미지 */}
      <div className="relative w-[280px] h-[280px]">
        {!correctionOn || !correctedSrc ? (
          <img
            ref={imgRef}
            src={`/ishihara/Ishihara_${plate.id}.${plate.ext ?? "jpg"}`}
            alt={`이시하라 플레이트 ${plate.id}`}
            width={280}
            height={280}
            className="rounded-full border object-cover"
            style={{ borderColor: "var(--border)", width: 280, height: 280 }}
            crossOrigin="anonymous"
          />
        ) : (
          <img
            src={correctedSrc}
            alt="AI 보정"
            width={280}
            height={280}
            className="rounded-full border object-cover"
            style={{ borderColor: "var(--color-brand)", width: 280, height: 280 }}
          />
        )}
        {correcting && (
          <div className="absolute inset-0 rounded-full bg-black/60 flex items-center justify-center">
            <p className="text-white text-sm">보정 중...</p>
          </div>
        )}
      </div>

      {/* AI 보정 토글 */}
      <button
        onClick={() => setCorrectionOn(!correctionOn)}
        disabled={!ready || correcting}
        className="px-5 py-2 rounded-full text-sm font-medium transition-colors disabled:opacity-40"
        style={correctionOn
          ? { background: "var(--color-brand)", color: "#fff" }
          : { background: "var(--bg-muted)", color: "var(--fg)" }
        }
      >
        AI 보정 {correctionOn ? "ON" : "OFF"}
      </button>

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
            {current + 1 < PLATES.length ? "다음" : "완료"}
          </button>
        </div>
        <button
          onClick={() => { setInput("모름"); submit(); }}
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
