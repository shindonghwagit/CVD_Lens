/**
 * 이시하라 도판 메타데이터 + 진단 로직 (단일 출처).
 *
 * 출처: 표준 이시하라 38판 세트의 답 키. 메타데이터는 판 1~17, 22~25를 포함하고
 * 실제 한 번의 참고 검사는 그중 시범 1 + 무작위 선별 5 + 분류 4 = 10판을 사용한다.
 *   - demo          : 모두 읽는 대조판(판 1 = 12).
 *   - screening     : 적록 선별. transformation(정상·적록이 다른 숫자) + vanishing(정상만 읽음).
 *   - classification: P/D 감별판(판 22~25). 정상·protan·deutan이 서로 다른 숫자를 읽는다.
 *
 * 결과 범위: 뚜렷한 신호 없음 / 적록 반응 신호 / protan형 / deutan형.
 *   - 이 검사는 임상 진단이 아닌 화면 기반 간이 선별이다.
 *   - 적록 선별 신호가 먼저 확인된 경우에만 분류판으로 반응 경향을 제시한다.
 *   - Tritan형(청황 계열)은 표준 이시하라 38판으로 선별할 수 없어 범위 밖이다.
 */
export type PlateKind = "demo" | "screening" | "classification";

export interface Plate {
  id: string;
  /** 정상 시각이 읽는 값. (vanishing 판은 정상만 읽고 적록은 못 읽음) */
  answer: string;
  /** 적록 색각이상이 읽는 값. transformation=다른 숫자, vanishing/hidden=null(못 읽음). */
  rgAnswer: string | null;
  /** classification 전용: protan형 반응에서 읽는 값. */
  protan?: string;
  /** classification 전용: deutan형 반응에서 읽는 값. */
  deutan?: string;
  kind: PlateKind;
  label: string;
  ext?: "png" | "jpg";
}

// 답 키(판 1~17, 22~25). 판 18~21(은닉숫자: 정상=nothing)은 자가검사 UX상 제외.
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
  { id: "22", answer: "26", rgAnswer: null, protan: "6", deutan: "2", kind: "classification", label: "적록 유형 감별" },
  { id: "23", answer: "42", rgAnswer: null, protan: "2", deutan: "4", kind: "classification", label: "적록 유형 감별" },
  { id: "24", answer: "35", rgAnswer: null, protan: "5", deutan: "3", kind: "classification", label: "적록 유형 감별" },
  { id: "25", answer: "96", rgAnswer: null, protan: "6", deutan: "9", kind: "classification", label: "적록 유형 감별" },
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
  const shuffle = <T,>(items: T[]): T[] => {
    const result = [...items];
    for (let i = result.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  };
  const selectedScreening = shuffle(screening).slice(0, SCREENING_PER_TEST);
  return [...demo, ...shuffle([...selectedScreening, ...classification])];
}

export type Diagnosis = "normal" | "protan" | "deutan" | "rg";

export interface DiagnosisResult {
  type: Diagnosis;
  demoPassed: boolean;
  screeningErrors: number;
  screeningTotal: number;
  /** 분류판을 각 유형으로 읽은 수(진단 근거·막대그래프용). */
  normalReads: number;
  protanVotes: number;
  deutanVotes: number;
  classTotal: number;
}

const norm = (s: string | null) => (s ?? "").trim();

/**
 * 적록 간이 선별: 선별판 오답 수 + 분류판 응답으로 반응 경향을 계산한다.
 * tritan은 범위 밖(판정하지 않음).
 */
export function diagnose(plates: Plate[], answers: (string | null)[]): DiagnosisResult {
  const demo = plates.map((p, i) => ({ p, i })).find(({ p }) => p.kind === "demo");
  const demoPassed = demo ? norm(answers[demo.i]) === demo.p.answer : false;
  const screening = plates.map((p, i) => ({ p, i })).filter(({ p }) => p.kind === "screening");
  const screeningErrors = screening.filter(({ p, i }) => norm(answers[i]) !== p.answer).length;
  const screeningTotal = screening.length;

  const classification = plates.map((p, i) => ({ p, i })).filter(({ p }) => p.kind === "classification");
  let protanVotes = 0;
  let deutanVotes = 0;
  let normalReads = 0;
  for (const { p, i } of classification) {
    const a = norm(answers[i]);
    if (a === p.answer) normalReads++;
    else if (p.protan && a === p.protan) protanVotes++;
    else if (p.deutan && a === p.deutan) deutanVotes++;
  }
  const classTotal = classification.length;

  let type: Diagnosis;
  // 표준 38판의 단축법은 모든 판을 정상적으로 읽은 경우에만 정상 근거로 삼고,
  // 불일치가 있으면 전체 검사를 권한다. 이 5판 간이 선별에서는 1~2개 불일치를
  // 판정 유보, 3개 이상을 뚜렷한 선별 신호로 보수적으로 취급한다.
  const screeningSignal = demoPassed && screeningErrors >= 3;
  const strongProtanPattern = protanVotes >= 3 && protanVotes - deutanVotes >= 2;
  const strongDeutanPattern = deutanVotes >= 3 && deutanVotes - protanVotes >= 2;

  if (demoPassed && screeningErrors === 0 && !strongProtanPattern && !strongDeutanPattern) type = "normal";
  else if (!screeningSignal) type = "rg"; // 대조판 실패 또는 소수 불일치: 판정 유보
  else if (strongProtanPattern) type = "protan";
  else if (strongDeutanPattern) type = "deutan";
  else type = "rg"; // 적록 신호는 있으나 P/D 반응이 일관되지 않음

  return { type, demoPassed, screeningErrors, screeningTotal, normalReads, protanVotes, deutanVotes, classTotal };
}

/** 간이 선별 결과 → 화면 라벨/색. */
export const DIAGNOSIS_META: Record<Diagnosis, { label: string; color: string; correctionType: "p" | "d" | null }> = {
  normal: { label: "뚜렷한 적록 색각이상 신호 없음", color: "#22c55e", correctionType: null },
  protan: { label: "Protan형(적색계) 반응 가능성", color: "#ef4444", correctionType: "p" },
  deutan: { label: "Deutan형(녹색계) 반응 가능성", color: "#16a34a", correctionType: "d" },
  rg: { label: "판정 유보 · 유형 확인 필요", color: "#f97316", correctionType: null },
};

export interface Bar {
  key: "normal" | "protan" | "deutan";
  label: string;
  color: string;
  value: number;
  max: number;
  note?: string;
}

/**
 * 막대그래프 데이터: 분류판에서 일반·Protan형·Deutan형 응답이 나온 횟수.
 * Tritan형은 표준 이시하라로 선별할 수 없어 이 그래프에서 제외한다.
 */
export function diagnosisBars(r: DiagnosisResult): Bar[] {
  return [
    { key: "normal", label: "일반 반응", color: "#64748b", value: r.normalReads, max: r.classTotal },
    { key: "protan", label: "Protan형", color: "#ef4444", value: r.protanVotes, max: r.classTotal },
    { key: "deutan", label: "Deutan형", color: "#16a34a", value: r.deutanVotes, max: r.classTotal },
  ];
}
