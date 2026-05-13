"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useSession } from "next-auth/react";
import { useModel } from "../context/ModelContext";
import { CVDType } from "../hooks/useCVDModel";

// 플레이트 정의
// targetCVD: 이 색각이상 타입을 가진 사람이 보기 어려운 플레이트
const PLATES = [
  { number: "12", targetCVD: "pd" as const, label: "제1·2색맹 감별" },
  { number: "8",  targetCVD: "pd" as const, label: "제1·2색맹 감별" },
  { number: "29", targetCVD: "pd" as const, label: "제1·2색맹 감별" },
  { number: "5",  targetCVD: "t"  as const, label: "제3색맹 감별" },
];

const SIZE = 300;

function hsl(h: number, s: number, l: number): string {
  return `hsl(${h},${s}%,${l}%)`;
}

// Canvas에 이시하라 유사 플레이트 생성
function generatePlate(
  canvas: HTMLCanvasElement,
  text: string,
  targetCVD: "pd" | "t",
  seed: number = 42
) {
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext("2d")!;

  // ── 1. 숫자 마스크 생성 ──────────────────────────
  const mask = document.createElement("canvas");
  mask.width = SIZE;
  mask.height = SIZE;
  const mCtx = mask.getContext("2d")!;
  mCtx.fillStyle = "#000";
  mCtx.fillRect(0, 0, SIZE, SIZE);
  mCtx.fillStyle = "#fff";
  mCtx.font = `bold ${text.length > 1 ? 110 : 140}px serif`;
  mCtx.textAlign = "center";
  mCtx.textBaseline = "middle";
  mCtx.fillText(text, SIZE / 2, SIZE / 2);
  const maskData = mCtx.getImageData(0, 0, SIZE, SIZE).data;

  // ── 2. 배경 칠하기 ──────────────────────────────
  ctx.fillStyle = "#d0d0d0";
  ctx.fillRect(0, 0, SIZE, SIZE);

  // 원형 클립
  ctx.save();
  ctx.beginPath();
  ctx.arc(SIZE / 2, SIZE / 2, SIZE / 2 - 2, 0, Math.PI * 2);
  ctx.clip();

  // ── 3. 랜덤 도트 그리기 ─────────────────────────
  const rng = mulberry32(seed); // 플레이트마다 다른 시드
  for (let i = 0; i < 2200; i++) {
    const x = rng() * SIZE;
    const y = rng() * SIZE;
    const r = 4 + rng() * 9;
    const px = Math.min(SIZE - 1, Math.max(0, Math.floor(x)));
    const py = Math.min(SIZE - 1, Math.max(0, Math.floor(y)));
    const isFigure = maskData[(py * SIZE + px) * 4] > 128;

    if (targetCVD === "pd") {
      // 제1·2색맹: 빨강-초록 혼동
      if (isFigure) {
        ctx.fillStyle = hsl(rng() * 30, 70 + rng() * 25, 38 + rng() * 18); // red-orange
      } else {
        ctx.fillStyle = hsl(80 + rng() * 40, 55 + rng() * 30, 38 + rng() * 18); // green-yellow
      }
    } else {
      // 제3색맹: 파랑-노랑 혼동
      if (isFigure) {
        ctx.fillStyle = hsl(210 + rng() * 50, 70 + rng() * 25, 38 + rng() * 18); // blue-purple
      } else {
        ctx.fillStyle = hsl(35 + rng() * 30, 75 + rng() * 20, 45 + rng() * 18); // yellow-orange
      }
    }

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

// 시드 기반 랜덤 (재현 가능한 플레이트)
function mulberry32(seed: number) {
  let s = seed;
  return () => {
    s |= 0; s = s + 0x6d2b79f5 | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ─────────────────────────────────────────────────────
// 메인 컴포넌트
// ─────────────────────────────────────────────────────
export default function IshiharaTest() {
  const { ready, error, infer } = useModel();
  const { data: session } = useSession();
  const [current, setCurrent] = useState(0);
  const [answers, setAnswers] = useState<(string | null)[]>(Array(PLATES.length).fill(null));
  const [seeds] = useState(() => Array.from({ length: PLATES.length }, () => Math.floor(Math.random() * 100000)));
  const [input, setInput] = useState("");
  const [correctionOn, setCorrectionOn] = useState(false);
  const [correctedSrc, setCorrectedSrc] = useState<string | null>(null);
  const [correcting, setCorrecting] = useState(false);
  const [done, setDone] = useState(false);

  const plateCanvasRef = useRef<HTMLCanvasElement>(null);
  const plate = PLATES[current];

  // 플레이트 생성
  useEffect(() => {
    if (!plateCanvasRef.current) return;
    generatePlate(plateCanvasRef.current, plate.number, plate.targetCVD, seeds[current]);
    setCorrectionOn(false);
    setCorrectedSrc(null);
    setInput("");
  }, [current, plate]);

  // 보정 적용
  const applyCorrection = useCallback(async () => {
    if (!ready || !plateCanvasRef.current) return;
    setCorrecting(true);

    const canvas = plateCanvasRef.current;
    const ctx = canvas.getContext("2d")!;
    const imageData = ctx.getImageData(0, 0, SIZE, SIZE);

    // CVD 타입 선택: pd → deuteranopia(d), t → tritanopia(t)
    const cvdType: CVDType = plate.targetCVD === "t" ? "t" : "d";
    const result = await infer(imageData, cvdType);

    const outCanvas = document.createElement("canvas");
    outCanvas.width = SIZE;
    outCanvas.height = SIZE;
    outCanvas.getContext("2d")!.putImageData(result, 0, 0);
    setCorrectedSrc(outCanvas.toDataURL());
    setCorrecting(false);
  }, [ready, infer, plate]);

  useEffect(() => {
    if (correctionOn && !correctedSrc) applyCorrection();
  }, [correctionOn, correctedSrc, applyCorrection]);

  // 답 제출
  const submit = async () => {
    const newAnswers = [...answers];
    newAnswers[current] = input.trim() || "모름";
    setAnswers(newAnswers);

    if (current + 1 >= PLATES.length) {
      // 결과 집계 후 저장
      const correct = newAnswers.filter((ans, i) => ans === PLATES[i].number).length;
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
  };

  function getDiagnosis(ans: (string | null)[]) {
    let pdWrong = 0, tWrong = 0;
    ans.forEach((a, i) => {
      if (a !== PLATES[i].number) {
        if (PLATES[i].targetCVD === "pd") pdWrong++;
        else tWrong++;
      }
    });
    if (pdWrong >= 2) return "제1·2색맹 가능성";
    if (tWrong >= 1)  return "제3색맹 가능성";
    return "정상";
  }

  const getResult = () => {
    const d = getDiagnosis(answers);
    if (d === "제1·2색맹 가능성") return "제1·2색맹(적록색각이상) 가능성이 있습니다.";
    if (d === "제3색맹 가능성")   return "제3색맹(청황색각이상) 가능성이 있습니다.";
    return "정상 색각으로 판단됩니다.";
  };

  if (done) {
    return (
      <div className="flex flex-col items-center gap-6 py-8">
        <div className="text-center">
          <p className="text-2xl font-bold mb-2">진단 완료</p>
          <p className="text-gray-300">{getResult()}</p>
          <p className="text-gray-500 text-xs mt-2">※ 이 결과는 참고용이며 정확한 진단은 안과 전문의에게 받으세요.</p>
        </div>

        <div className="w-full rounded-xl p-4 flex flex-col gap-2" style={{ background: "var(--bg-muted)" }}>
          {PLATES.map((p, i) => (
            <div key={i} className="flex justify-between text-sm">
              <span className="text-gray-400">플레이트 {i + 1} (정답: {p.number})</span>
              <span className={answers[i] === p.number ? "text-green-400" : "text-red-400"}>
                내 답: {answers[i] ?? "-"} {answers[i] === p.number ? "✓" : "✗"}
              </span>
            </div>
          ))}
        </div>

        <button
          onClick={() => { setCurrent(0); setAnswers(Array(PLATES.length).fill(null)); setDone(false); window.location.reload(); }}
          className="px-6 py-2 rounded-full bg-blue-600 hover:bg-blue-500 text-sm font-medium"
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
            className="bg-blue-600 h-1.5 rounded-full transition-all"
            style={{ width: `${((current) / PLATES.length) * 100}%` }}
          />
        </div>
        <span className="text-gray-400 text-sm">{current + 1} / {PLATES.length}</span>
      </div>

      <p className="text-gray-300 text-sm">{plate.label}</p>

      {/* 플레이트 + 보정 토글 */}
      <div className="relative">
        <canvas
          ref={plateCanvasRef}
          width={SIZE}
          height={SIZE}
          className="rounded-full border border-gray-700"
          style={{ display: correctionOn && correctedSrc ? "none" : "block", width: 280, height: 280 }}
        />
        {correctionOn && correctedSrc && (
          <img
            src={correctedSrc}
            alt="corrected"
            className="rounded-full border border-blue-500"
            style={{ width: 280, height: 280 }}
          />
        )}
        {correcting && (
          <div className="absolute inset-0 rounded-full bg-black/60 flex items-center justify-center">
            <p className="text-white text-sm">보정 중...</p>
          </div>
        )}
      </div>

      {/* 보정 토글 */}
      <button
        onClick={() => setCorrectionOn(!correctionOn)}
        disabled={!ready || correcting}
        className={`px-5 py-2 rounded-full text-sm font-medium transition-colors disabled:opacity-40 ${
          correctionOn ? "bg-brand text-white" : "hover:opacity-80" 
        }`} style={!correctionOn ? { background: "var(--bg-muted)", color: "var(--fg)" } : undefined}
      >
        {correctionOn ? "AI 보정 ON" : "AI 보정 OFF"}
      </button>

      {/* 답 입력 */}
      <div className="flex flex-col items-center gap-3 w-full max-w-xs">
        <p className="text-sm text-gray-400">보이는 숫자를 입력하세요</p>
        <div className="flex gap-2 w-full">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
            placeholder="숫자 입력 (모르면 빈칸)"
            className="flex-1 rounded-lg px-3 py-2 text-sm focus:outline-none border" style={{ background: "var(--bg-muted)", borderColor: "var(--border-strong)", color: "var(--fg)" }}
          />
          <button
            onClick={submit}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-medium"
          >
            {current + 1 < PLATES.length ? "다음" : "완료"}
          </button>
        </div>
        <button onClick={() => { setInput(""); submit(); }} className="text-xs text-gray-600 hover:text-gray-400">
          잘 모르겠어요 (건너뛰기)
        </button>
      </div>

      {!ready && <p className="text-yellow-400 text-sm">모델 로딩 중...</p>}
      {error && <p className="text-red-400 text-sm">오류: {error}</p>}
    </div>
  );
}
