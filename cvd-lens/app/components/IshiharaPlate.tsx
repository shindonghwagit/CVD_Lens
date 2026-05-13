"use client";

import { useEffect, useRef } from "react";

type Variant = "pd" | "t";

const PALETTES = {
  pd: {
    original: {
      figure: ["#d85b4a", "#c94a3a", "#e4735a", "#bf553c", "#d65d44"],
      ground: ["#a6a25a", "#b8a854", "#9d9c5e", "#c1ad5f", "#aba95d"],
      bg: "#e6e4d9",
    },
    corrected: {
      figure: ["#e22b36", "#c8241c", "#f04540", "#d21a2a", "#e63238"],
      ground: ["#4d8a3c", "#3f7a31", "#5a9a48", "#4c8f40", "#5fa052"],
      bg: "#eaeadf",
    },
  },
  t: {
    original: {
      figure: ["#7a8a98", "#8a99a7", "#829a9e", "#6e8797"],
      ground: ["#c6b578", "#b9a66c", "#d0bd80", "#bda964"],
      bg: "#e9e3d2",
    },
    corrected: {
      figure: ["#2e5a9b", "#285291", "#3768a8", "#254a86"],
      ground: ["#e3b839", "#d9ad2f", "#edc34a", "#cfa220"],
      bg: "#eae3d0",
    },
  },
};

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function drawPlate(
  canvas: HTMLCanvasElement,
  opts: { text: string; variant: Variant; corrected: boolean; seed: number; size: number }
) {
  const { text, variant, corrected, seed, size } = opts;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const pal = PALETTES[variant][corrected ? "corrected" : "original"];

  const mask = document.createElement("canvas");
  mask.width = size;
  mask.height = size;
  const mCtx = mask.getContext("2d")!;
  mCtx.fillStyle = "#000";
  mCtx.fillRect(0, 0, size, size);
  mCtx.fillStyle = "#fff";
  mCtx.font = `bold ${text.length > 1 ? size * 0.36 : size * 0.46}px "Instrument Serif", serif`;
  mCtx.textAlign = "center";
  mCtx.textBaseline = "middle";
  mCtx.fillText(text, size / 2, size / 2 + size * 0.02);
  const md = mCtx.getImageData(0, 0, size, size).data;

  ctx.fillStyle = pal.bg;
  ctx.fillRect(0, 0, size, size);
  ctx.save();
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 2, 0, Math.PI * 2);
  ctx.clip();

  const rng = mulberry32(seed);
  const dotCount = Math.floor(size * 7);
  for (let i = 0; i < dotCount; i++) {
    const x = rng() * size;
    const y = rng() * size;
    const r = 3 + rng() * (size * 0.022);
    const px = Math.max(0, Math.min(size - 1, Math.floor(x)));
    const py = Math.max(0, Math.min(size - 1, Math.floor(y)));
    const isFigure = md[(py * size + px) * 4] > 128;
    const palette = isFigure ? pal.figure : pal.ground;
    ctx.fillStyle = palette[Math.floor(rng() * palette.length)];
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

export default function IshiharaPlateCanvas({
  text,
  variant = "pd",
  corrected = false,
  seed = 42,
  size = 320,
  className,
}: {
  text: string;
  variant?: Variant;
  corrected?: boolean;
  seed?: number;
  size?: number;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (ref.current) drawPlate(ref.current, { text, variant, corrected, seed, size });
  }, [text, variant, corrected, seed, size]);
  return <canvas ref={ref} className={className} />;
}