/**
 * ishiharaGen — 브라우저 canvas로 이시하라 도판을 **절차적 생성**한다.
 *
 * 고정 이미지(/ishihara/*.jpg)는 매번 같은 숫자만 나오므로, 검사할 때마다
 * 랜덤 숫자로 새 도판을 그려 "매번 똑같은 숫자" 문제를 없앤다.
 *
 * 정직성(honest red-green screening) 유지:
 *   - screening 도판 = 적록 혼동선 배색(figure=주황/빨강, ground=초록/올리브).
 *     정상시각은 명도가 아닌 색상(hue)으로 숫자를 읽고, 적록 색각이상은 흐려진다.
 *   - demo(대조) 도판 = figure/ground를 **명도**로 구분 → 색각과 무관하게 모두 읽음.
 *   - P/D 감별이나 tritan(청색) 주장은 하지 않는다. 결과는 적록 선별 수준까지만.
 *
 * 배치: jittered-grid 1-pass(거부 샘플링 없음) → 메인스레드에서 즉각적.
 */
import type { Plate } from "./ishiharaPlates";

export interface GenPlate extends Plate {
  /** 절차적으로 렌더된 도판 PNG (data URL). plateSrc 대신 이걸 쓴다. */
  dataUrl: string;
}

const SIZE = 420;
const R = 198;
const C = SIZE / 2;
const CELL = 13;

/** HSV(0..1) → [r,g,b] 0..255 */
function hsv2rgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  let r = 0, g = 0, b = 0;
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    case 5: r = v; g = p; b = q; break;
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

const rand = (lo: number, hi: number) => lo + Math.random() * (hi - lo);

/** 숫자 마스크를 그려 각 픽셀의 figure 여부(알파>128)를 담은 Uint8Array 반환. */
function buildMask(text: string): Uint8ClampedArray {
  const m = document.createElement("canvas");
  m.width = SIZE;
  m.height = SIZE;
  const ctx = m.getContext("2d")!;
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.fillStyle = "#000";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // 폭이 원 안에 들도록 폰트 크기 자동 축소.
  let fs = 240;
  ctx.font = `bold ${fs}px Arial, sans-serif`;
  const maxW = R * 1.7;
  const w = ctx.measureText(text).width;
  if (w > maxW) {
    fs = Math.floor((fs * maxW) / w);
    ctx.font = `bold ${fs}px Arial, sans-serif`;
  }
  ctx.fillText(text, C, C + fs * 0.02);
  return ctx.getImageData(0, 0, SIZE, SIZE).data;
}

/**
 * 도판 1장 렌더 → data URL.
 * kind="demo" : 명도 대비(모두 읽음) / kind="screening" : 적록 혼동선(정상만 읽음).
 */
function renderPlate(text: string, kind: "demo" | "screening"): string {
  const mask = buildMask(text);
  const cv = document.createElement("canvas");
  cv.width = SIZE;
  cv.height = SIZE;
  const ctx = cv.getContext("2d")!;

  // 원 안쪽만 그리도록 클립 + 배경(off-white).
  ctx.save();
  ctx.beginPath();
  ctx.arc(C, C, R, 0, Math.PI * 2);
  ctx.clip();
  ctx.fillStyle = "#f6f4ef";
  ctx.fillRect(0, 0, SIZE, SIZE);

  for (let gy = 0; gy < SIZE; gy += CELL) {
    for (let gx = 0; gx < SIZE; gx += CELL) {
      const r = rand(CELL * 0.3, CELL * 0.48);
      const x = gx + CELL / 2 + rand(-CELL * 0.22, CELL * 0.22);
      const y = gy + CELL / 2 + rand(-CELL * 0.22, CELL * 0.22);
      const dx = x - C, dy = y - C;
      if (dx * dx + dy * dy > (R - r) * (R - r)) continue;

      const idx = (Math.round(y) * SIZE + Math.round(x)) * 4 + 3;
      const fig = mask[idx] > 128;

      let h: number, s: number, v: number;
      if (kind === "demo") {
        // 대조: 밝은 주황 숫자 vs 중성 회색 배경 → 명도로 구분(모두 읽음).
        if (fig) { h = rand(20, 38) / 360; s = rand(0.75, 0.9); v = rand(0.9, 0.99); }
        else { h = rand(20, 45) / 360; s = rand(0.04, 0.1); v = rand(0.62, 0.74); }
      } else {
        // 적록 혼동선: 주황/빨강 숫자 vs 초록/올리브 배경(명도 겹침 → 색상만이 단서).
        if (fig) { h = rand(8, 34) / 360; s = rand(0.55, 0.85); v = rand(0.6, 0.92); }
        else { h = rand(74, 132) / 360; s = rand(0.32, 0.62); v = rand(0.58, 0.9); }
      }
      const [rr, gg, bb] = hsv2rgb(h, s, v);
      ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
  return cv.toDataURL("image/png");
}

/** 랜덤 2자리 숫자(10~99). */
const randomNumber = () => String(Math.floor(rand(10, 100)));

/**
 * 검사용 도판 세트 생성: 대조 1장 + 적록 선별 4장, 각각 랜덤 숫자.
 * 클라이언트에서만 호출(canvas 필요). id는 렌더마다 유일.
 */
export function generatePlateSet(): GenPlate[] {
  const stamp = Date.now().toString(36);
  const mk = (i: number, kind: "demo" | "screening"): GenPlate => {
    const answer = randomNumber();
    return {
      id: `${stamp}-${i}`,
      answer,
      altAnswer: null,
      series: kind === "demo" ? "control" : "red-green",
      purpose: kind === "demo" ? "demo" : "screening",
      diagnostic: kind === "screening",
      label: kind === "demo" ? "모든 분이 읽을 수 있는 대조 도판" : "적록 선별",
      dataUrl: renderPlate(answer, kind),
    };
  };
  return [mk(0, "demo"), ...[1, 2, 3, 4].map((i) => mk(i, "screening"))];
}
