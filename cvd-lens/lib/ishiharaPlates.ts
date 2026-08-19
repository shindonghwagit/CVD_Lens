/**
 * 이시하라 도판 메타데이터 (단일 출처).
 *
 * 표준 이시하라 도판(1~15)은 **적록(red-green) 선별** 검사다. tritan(청색) 감별력은
 * 검증되지 않았으므로 진단 풀에서 제외하고, 계열도 "unclassified"로 둔다(추측 금지).
 * P vs D(적색맹 vs 녹색맹)를 이 도판들로 확정할 수 없으므로 어떤 도판도 P/D로 분류하지
 * 않는다 — 결과는 "적록 계열" 선별 수준까지만 주장한다(honest red-green screening, e2bc18a).
 *
 * series  : 오답이 시사하는 색 계열. control(대조) / red-green(적록) / unclassified(미분류).
 * purpose : 도판 기능. demo(시연·대조) / screening(선별=vanishing) / transformation(변환).
 *   - transformation = 정상과 색각이상자가 서로 다른 숫자를 읽는 도판(altAnswer 존재).
 *   - screening      = 정상은 읽고 색각이상자는 못 읽는 vanishing 도판.
 * answer/altAnswer 는 앱 기존 값을 그대로 보존(변경 금지).
 */
export type PlateSeries = "control" | "red-green" | "unclassified";
export type PlatePurpose = "demo" | "screening" | "transformation";

export interface Plate {
  id: string;
  answer: string;
  altAnswer: string | null; // 색각이상자가 읽는 값 (없으면 null)
  series: PlateSeries;
  purpose: PlatePurpose;
  /** 진단 풀 포함 여부. 표준 검증된 적록 도판만 true. */
  diagnostic: boolean;
  label: string;
  ext?: "png" | "jpg";
}

export const PLATES: Plate[] = [
  { id: "01", answer: "74", altAnswer: "21", series: "control", purpose: "demo", diagnostic: false, label: "모든 분이 읽을 수 있는 대조 도판" },
  { id: "03", answer: "16", altAnswer: null, series: "red-green", purpose: "screening", diagnostic: true, label: "적록 선별" },
  { id: "04", answer: "2", altAnswer: null, series: "red-green", purpose: "screening", diagnostic: true, label: "적록 선별" },
  { id: "05", answer: "29", altAnswer: null, series: "red-green", purpose: "screening", diagnostic: true, label: "적록 선별" },
  { id: "06", answer: "7", altAnswer: null, series: "red-green", purpose: "screening", diagnostic: true, label: "적록 선별" },
  { id: "07", answer: "45", altAnswer: null, series: "red-green", purpose: "screening", diagnostic: true, label: "적록 선별" },
  { id: "08", answer: "5", altAnswer: "2", series: "red-green", purpose: "transformation", diagnostic: true, label: "적록 변환(정상 5 · 색각이상 2)" },
  { id: "09", answer: "97", altAnswer: null, series: "red-green", purpose: "screening", diagnostic: true, label: "적록 선별" },
  { id: "10", answer: "8", altAnswer: null, series: "red-green", purpose: "screening", diagnostic: true, label: "적록 선별" },
  // 11: 청색 계열로 표시돼 있으나 표준 이시하라의 tritan 감별력 미검증 → 계열 미분류·진단 제외.
  { id: "11", answer: "42", altAnswer: null, series: "unclassified", purpose: "screening", diagnostic: false, label: "미분류 (참고용, 진단 제외)" },
  { id: "12", answer: "3", altAnswer: null, series: "red-green", purpose: "screening", diagnostic: true, label: "적록 선별" },
  { id: "13", answer: "42", altAnswer: null, series: "red-green", purpose: "screening", diagnostic: true, label: "적록 선별" },
  { id: "14", answer: "27", altAnswer: null, series: "red-green", purpose: "screening", diagnostic: true, label: "적록 선별" },
  { id: "15", answer: "12", altAnswer: null, series: "red-green", purpose: "screening", diagnostic: true, label: "적록 선별" },
];

export const plateSrc = (p: Plate) => `/ishihara/Ishihara_${p.id}.${p.ext ?? "jpg"}`;

/** 결과 막대그래프용 기능 카테고리(적록 원칙 준수 — R/G/B 축으로 쪼개지 않음). */
export const CATEGORIES: { key: PlatePurpose; label: string; color: string }[] = [
  { key: "demo", label: "대조", color: "#64748b" },
  { key: "screening", label: "적록 선별", color: "#2a5fd9" },
  { key: "transformation", label: "적록 변환", color: "#7c3aed" },
];

/** 검사에 쓸 도판 뽑기: 대조 1장 + 진단(적록) 도판 랜덤 4장. */
export function pickPlates(): Plate[] {
  const demo = PLATES.filter((p) => p.purpose === "demo");
  const diagnostic = PLATES.filter((p) => p.diagnostic);
  const shuffled = [...diagnostic].sort(() => Math.random() - 0.5);
  return [...demo, ...shuffled.slice(0, 4)];
}

export interface CategoryScore {
  key: PlatePurpose;
  label: string;
  color: string;
  correct: number;
  total: number;
}

/** 카테고리별 정답 수 집계 (검사에 등장한 카테고리만). */
export function categoryScores(plates: Plate[], answers: (string | null)[]): CategoryScore[] {
  return CATEGORIES.map((c) => {
    const idxs = plates.map((p, i) => ({ p, i })).filter(({ p }) => p.purpose === c.key);
    const correct = idxs.filter(({ p, i }) => answers[i] === p.answer).length;
    return { key: c.key, label: c.label, color: c.color, correct, total: idxs.length };
  }).filter((c) => c.total > 0);
}

/**
 * 유형 추정 — 적록 선별 원칙 유지. 적록(diagnostic) 도판의 오답 비율만 본다.
 * P/D 확정이나 tritan 진단은 하지 않는다. 반환: "red-green" | "normal".
 */
export function estimateType(plates: Plate[], answers: (string | null)[]): "red-green" | "normal" {
  const diag = plates.map((p, i) => ({ p, i })).filter(({ p }) => p.diagnostic);
  const wrong = diag.filter(({ p, i }) => answers[i] !== p.answer).length;
  const threshold = Math.max(1, Math.round(diag.length / 2));
  return wrong >= threshold ? "red-green" : "normal";
}
