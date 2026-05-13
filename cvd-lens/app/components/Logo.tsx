import * as React from "react";

export default function Logo({
  size = 28,
  showWordmark = false,
  className,
}: {
  size?: number;
  showWordmark?: boolean;
  className?: string;
}) {
  return (
    <span className={`inline-flex items-center gap-2.5 ${className ?? ""}`}>
      <svg
        viewBox="0 0 40 40"
        width={size}
        height={size}
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* outer lens ring */}
        <circle cx="20" cy="20" r="17" stroke="var(--color-brand)" strokeWidth="2" />
        {/* inner eye outline */}
        <path
          d="M6 20c2.6-5 7.8-8 14-8s11.4 3 14 8c-2.6 5-7.8 8-14 8s-11.4-3-14-8Z"
          fill="var(--color-brand-soft)"
          stroke="var(--color-brand-ink)"
          strokeWidth="1.3"
        />
        {/* pupil (brand ink) */}
        <circle cx="20" cy="20" r="4" fill="var(--color-brand-ink)" />
        {/* highlight */}
        <circle cx="18.6" cy="18.6" r="1.2" fill="#ffffff" />
        {/* RGB spectrum dots — the "correction" */}
        <circle cx="34" cy="10" r="2" fill="#d85b4a" />
        <circle cx="37" cy="22" r="1.6" fill="#2f9e6b" />
        <circle cx="31" cy="33" r="1.6" fill="#2a5fd9" />
      </svg>
      {showWordmark && (
        <span className="flex flex-col leading-none">
          <span className="font-semibold tracking-tight text-[15px]">CVDLens</span>
          <span
            className="font-mono text-[9px] tracking-[0.14em] mt-0.5"
            style={{ color: "var(--fg-subtle)" }}
          >
            COLOR VISION ASSIST
          </span>
        </span>
      )}
    </span>
  );
}