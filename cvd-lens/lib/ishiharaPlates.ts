/**
 * 이시하라 도판 메타데이터 + 진단 로직 (단일 출처).
 *
 * 출처: 표준 이시하라 38판 세트(공식 답 키). 이 프로젝트는 판 1~24를 사용한다.
 *   - demo          : 모두 읽는 대조판(판 1 = 12).
 *   - screening     : 적록 선별. transformation(정상·적록이 다른 숫자) + vanishing(정상만 읽음).
 *   - classification: P/D 감별판(판 22~25). 정상·protan·deutan이 서로 다른 숫자를 읽는다.
 *
 * 진단 범위: 정상 / 적색맹(protan) / 녹색맹(deutan).
 *   - 적록 선별로 이상 여부를 감지하고, 분류판 응답으로 protan vs deutan을 가른다.
 *   - 청색맹(tritan)은 표준 이시하라 38판에 판이 없어 진단 불가 → 범위 밖(별도 판 필요).
 */
export type PlateKind = "demo" | "screening" | "classification";

export interface Plate {
  id: string;
  /** 정상 시각이 읽는 값. (vanishing 판은 정상만 읽고 적록은 못 읽음) */
  answer: string;
  /** 적록 색각이상이 읽는 값. transformation=다른 숫자, vanishing/hidden=null(못 읽음). */
  rgAnswer: string | null;
  /** classification 전용: protan(적색맹)이 읽는 값. */
  protan?: string;
  /** classification 전용: deutan(녹색맹)이 읽는 값. */
  deutan?: string;
  kind: PlateKind;
  label: string;
  ext?: "png" | "jpg";
}

// 공식 답 키(판 1~24). 판 18~21(은닉숫자: 정상=nothing)은 자가검사 UX상 제외.
export const PLATES: Plate[] = [
  { id: "01", answer: "12", rgAnswer: "12", kind: "demo", label: "모든 분이 읽을 수 있는 대조 도판" },
  // transformation — 정상과 적록이 서로 다른 숫자를 읽음
  { id: "02", answer: "8", rgAnswer: "3", kind: "screening", label: "적록 선별" },
  { id: "03", answer: "6", rgAnswer: "5", kind: "screening", label: "적록 선별" },
  { id: "04", answer: "29", rgAnswer: "70", kind: "screening", label: "적록 선별" },
  { id: "05", answer: "57", rgAnswer: "35", kind: "screening", label: "적록 선별" },
  { id: "06", answer: "5", rgAnswer: "2", kind: "screening", label: "적록 선별" },
  { id: "07", answer: "3", rgAnswer: "5", kind: "screening", label: "적록 선별" },
  { id: "08", answer: "15", rgAnswer: "17", kind: "screening", label: "적록 선별" },
  { id: "09", answer: "74", rgAnswer: "21", kind: "screening", label: "적록 선별" },
  // vanishing — 정상만 읽고 적록은 못 읽음(오답/모름)
  { id: "10", answer: "2", rgAnswer: null, kind: "screening", label: "적록 선별" },
  { id: "11", answer: "6", rgAnswer: null, kind: "screening", label: "적록 선별" },
  { id: "12", answer: "97", rgAnswer: null, kind: "screening", label: "적록 선별" },
  { id: "13", answer: "45", rgAnswer: null, kind: "screening", label: "적록 선별" },
  { id: "14", answer: "5", rgAnswer: null, kind: "screening", label: "적록 선별" },
  { id: "15", answer: "7", rgAnswer: null, kind: "screening", label: "적록 선별" },
  { id: "16", answer: "16", rgAnswer: null, kind: "screening", label: "적록 선별" },
  { id: "17", answer: "73", rgAnswer: null, kind: "screening", label: "적록 선별" },
  // classification — protan vs deutan 감별
  { id: "22", answer: "26", rgAnswer: null, protan: "6", deutan: "2", kind: "classification", label: "유형 감별 (정상 26)" },
  { id: "23", answer: "42", rgAnswer: null, protan: "2", deutan: "4", kind: "classification", label: "유형 감별 (정상 42)" },
  { id: "24", answer: "35", rgAnswer: null, protan: "5", deutan: "3", kind: "classification", label: "유형 감별 (정상 35)" },
  { id: "25", answer: "96", rgAnswer: null, protan: "6", deutan: "9", kind: "classification", label: "유형 감별 (정상 96)" },
];

export const plateSrc = (p: Plate) => `/ishihara/Ishihara_${p.id}.${p.ext ?? "jpg"}`;

const SCREENING_PER_TEST = 5;

/**
 * 검사용 도판 뽑기: 대조 1 + 적록 선별 랜덤 N + 분류판 전체.
 * 분류판은 항상 포함(P/D 감별의 유일한 근거).
 */
export function pickPlates(): Plate[] {
  const demo = PLATES.filter((p) => p.kind === "demo");
  const screening = PLATES.filter((p) => p.kind === "screening");
  const classification = PLATES.filter((p) => p.kind === "classification");
  const shuffled = [...screening].sort(() => Math.random() - 0.5).slice(0, SCREENING_PER_TEST);
  return [...demo, ...shuffled, ...classification];
}

export type Diagnosis = "normal" | "protan" | "deutan" | "rg";

export interface DiagnosisResult {
  type: Diagnosis;
  screeningErrors: number;
  screeningTotal: number;
  /** 분류판을 각 유형으로 읽은 수(진단 근거·막대그래프용). */
  protanVotes: number;
  deutanVotes: number;
  classTotal: number;
}

const norm = (s: string | null) => (s ?? "").trim();

/**
 * 진단: 적록 선별 오답률 + 분류판 응답으로 정상/적색맹/녹색맹 판정.
 * tritan은 범위 밖(판정하지 않음).
 */
export function diagnose(plates: Plate[], answers: (string | null)[]): DiagnosisResult {
  const screening = plates.map((p, i) => ({ p, i })).filter(({ p }) => p.kind === "screening");
  const screeningErrors = screening.filter(({ p, i }) => norm(answers[i]) !== p.answer).length;
  const screeningTotal = screening.length;

  const classification = plates.map((p, i) => ({ p, i })).filter(({ p }) => p.kind === "classification");
  let protanVotes = 0;
  let deutanVotes = 0;
  for (const { p, i } of classification) {
    const a = norm(answers[i]);
    if (p.protan && a === p.protan) protanVotes++;
    else if (p.deutan && a === p.deutan) deutanVotes++;
  }
  const classTotal = classification.length;

  const screenThreshold = Math.max(2, Math.round(screeningTotal / 2));
  const deficient = screeningErrors >= screenThreshold || protanVotes + deutanVotes >= 2;

  let type: Diagnosis;
  if (!deficient) type = "normal";
  else if (protanVotes > deutanVotes) type = "protan";
  else if (deutanVotes > protanVotes) type = "deutan";
  else type = "rg"; // 적록 이상은 있으나 P/D 미분류

  return { type, screeningErrors, screeningTotal, protanVotes, deutanVotes, classTotal };
}

/** 진단 유형 → 화면 라벨/색. */
export const DIAGNOSIS_META: Record<Diagnosis, { label: string; color: string; correctionType: "p" | "d" | null }> = {
  normal: { label: "정상 (색각이상 징후 없음)", color: "#22c55e", correctionType: null },
  protan: { label: "적색맹(protan) 가능성", color: "#ef4444", correctionType: "p" },
  deutan: { label: "녹색맹(deutan) 가능성", color: "#22c55e", correctionType: "d" },
  rg: { label: "적록 색각이상 가능성 (유형 미분류)", color: "#f97316", correctionType: "d" },
};

export interface Bar {
  key: "protan" | "deutan" | "tritan";
  label: string;
  color: string;
  value: number;
  max: number;
  note?: string;
}

/** 막대그래프 데이터: 유형별 색각이상 신호(분류판 반응 수). tritan은 판 미보유 → 준비 중. */
export function diagnosisBars(r: DiagnosisResult): Bar[] {
  return [
    { key: "protan", label: "적색맹", color: "#ef4444", value: r.protanVotes, max: r.classTotal },
    { key: "deutan", label: "녹색맹", color: "#22c55e", value: r.deutanVotes, max: r.classTotal },
    { key: "tritan", label: "청색맹", color: "#3b82f6", value: 0, max: r.classTotal, note: "검사 판 준비 중" },
  ];
}
