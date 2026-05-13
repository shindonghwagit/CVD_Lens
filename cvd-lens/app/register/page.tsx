"use client";

import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Logo from "../components/Logo";

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const res = await fetch("/api/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, name }),
    });
    setLoading(false);
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      setError(j.error || "회원가입에 실패했습니다.");
      return;
    }
    router.push("/login");
  };

  return (
    <div className="min-h-[calc(100vh-80px)] grid md:grid-cols-2">
      <div className="hidden md:flex flex-col justify-between p-12" style={{ background: "var(--bg-muted)" }}>
        <Logo size={40} showWordmark />
        <div>
          <p className="font-mono text-[11px] tracking-[0.16em] uppercase mb-4" style={{ color: "var(--fg-subtle)" }}>
            CREATE ACCOUNT
          </p>
          <h2 className="font-serif text-[44px] leading-[1.05] tracking-[-0.03em] mb-4">
            당신의 눈에,<br />맞춤 보정.
          </h2>
          <p className="text-[15px] max-w-[340px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>
            첫 진단 후 결과에 맞춘 AI 보정 프로필이 자동 생성됩니다.
          </p>
        </div>
        <div className="font-mono text-[11px] tracking-[0.08em]" style={{ color: "var(--fg-subtle)" }}>v0.3 · alpha</div>
      </div>

      <div className="flex items-center justify-center p-8 md:p-12">
        <form onSubmit={onSubmit} className="w-full max-w-[360px] flex flex-col gap-5">
          <div>
            <p className="font-mono text-[11px] tracking-[0.16em] uppercase mb-2" style={{ color: "var(--fg-subtle)" }}>SIGN UP</p>
            <h1 className="font-serif text-[32px] tracking-[-0.02em]">회원가입</h1>
          </div>

          <label className="flex flex-col gap-1.5">
            <span className="font-mono text-[11px] tracking-[0.08em]" style={{ color: "var(--fg-muted)" }}>NAME</span>
            <input value={name} onChange={(e) => setName(e.target.value)} required
              className="px-3.5 py-3 rounded-lg border text-[14px] focus:outline-none focus:ring-2 focus:ring-brand/30 transition"
              style={{ background: "var(--bg-elevated)", borderColor: "var(--border-strong)", color: "var(--fg)" }} />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="font-mono text-[11px] tracking-[0.08em]" style={{ color: "var(--fg-muted)" }}>EMAIL</span>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required
              className="px-3.5 py-3 rounded-lg border text-[14px] focus:outline-none focus:ring-2 focus:ring-brand/30 transition"
              style={{ background: "var(--bg-elevated)", borderColor: "var(--border-strong)", color: "var(--fg)" }} />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="font-mono text-[11px] tracking-[0.08em]" style={{ color: "var(--fg-muted)" }}>PASSWORD</span>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={6}
              className="px-3.5 py-3 rounded-lg border text-[14px] focus:outline-none focus:ring-2 focus:ring-brand/30 transition"
              style={{ background: "var(--bg-elevated)", borderColor: "var(--border-strong)", color: "var(--fg)" }} />
          </label>

          {error && <div className="text-[13px] px-3 py-2 rounded-lg" style={{ background: "#fdecea", color: "#b23a2a" }}>{error}</div>}

          <button type="submit" disabled={loading}
            className="w-full py-3 rounded-full bg-brand hover:bg-brand-ink text-white text-sm font-medium transition-colors disabled:opacity-60">
            {loading ? "생성 중…" : "계정 만들기"}
          </button>

          <p className="text-center text-[13px]" style={{ color: "var(--fg-muted)" }}>
            이미 계정이 있으신가요? <Link href="/login" className="text-brand hover:underline">로그인</Link>
          </p>
        </form>
      </div>
    </div>
  );
}