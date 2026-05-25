"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export default function HeroVisual() {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [pct, setPct] = useState(48);
  const [dragging, setDragging] = useState(false);

  const onMove = useCallback((clientX: number) => {
    if (!wrapRef.current) return;
    const r = wrapRef.current.getBoundingClientRect();
    setPct(Math.max(2, Math.min(98, ((clientX - r.left) / r.width) * 100)));
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const mv = (e: MouseEvent | TouchEvent) => {
      const x = "touches" in e ? e.touches[0].clientX : e.clientX;
      onMove(x);
    };
    const up = () => setDragging(false);
    window.addEventListener("mousemove", mv);
    window.addEventListener("mouseup", up);
    window.addEventListener("touchmove", mv);
    window.addEventListener("touchend", up);
    return () => {
      window.removeEventListener("mousemove", mv);
      window.removeEventListener("mouseup", up);
      window.removeEventListener("touchmove", mv);
      window.removeEventListener("touchend", up);
    };
  }, [dragging, onMove]);

  return (
    <div
      ref={wrapRef}
      className="relative w-full aspect-[5/4] rounded-[28px] overflow-hidden border shadow-lift select-none cursor-ew-resize"
      style={{ borderColor: "var(--border)", background: "var(--bg-elevated)" }}
      onMouseDown={(e) => { setDragging(true); onMove(e.clientX); }}
      onTouchStart={(e) => { setDragging(true); onMove(e.touches[0].clientX); }}
    >
      <span
        className="absolute top-4 left-4 px-2.5 py-1.5 rounded-full text-[10px] tracking-[0.12em] uppercase font-mono z-10"
        style={{ background: "rgba(255,255,255,0.9)", color: "#0d1b2a" }}
      >
        원본 · as seen
      </span>
      <span
        className="absolute top-4 right-4 px-2.5 py-1.5 rounded-full text-[10px] tracking-[0.12em] uppercase font-mono text-white z-10"
        style={{ background: "var(--color-brand)" }}
      >
        보정 · corrected
      </span>

      {/* Corrected (right side, full width with filter) */}
      <img
        src="/ishihara/Ishihara_01.jpg"
        alt="corrected"
        className="absolute inset-0 w-full h-full object-cover"
        style={{ filter: "saturate(2.2) contrast(1.15) brightness(1.05)" }}
        draggable={false}
      />

      {/* Original (left side, same size, clipped via clipPath) */}
      <img
        src="/ishihara/Ishihara_01.jpg"
        alt="original"
        className="absolute inset-0 w-full h-full object-cover"
        style={{ clipPath: `polygon(0 0, ${pct}% 0, ${pct}% 100%, 0 100%)` }}
        draggable={false}
      />

      {/* Divider */}
      <div
        className="absolute top-0 bottom-0 w-[2px] bg-white z-20"
        style={{ left: `${pct}%`, boxShadow: "0 0 0 1px rgba(13,27,42,.2)" }}
      >
        <div
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-11 h-11 rounded-full bg-white flex items-center justify-center shadow-lift"
          style={{ color: "var(--color-brand)" }}
        >
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M8 9l-4 3 4 3" />
            <path d="M16 9l4 3-4 3" />
          </svg>
        </div>
      </div>
    </div>
  );
}
