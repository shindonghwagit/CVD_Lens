"use client";

import Link from "next/link";
import { useState } from "react";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";
import Logo from "../components/Logo";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const res = await signIn("credentials", { email, password, redirect: false });
    setLoading(false);
    if (res?.error) setError("이메일 또는 비밀번호가 올바르지 않습니다.");
    else router.push("/");
  };

  return (
    <div className="min-h-[calc(100vh-80px)] grid md:grid-cols-2">
      {/* Left: brand side */}
      <div
        className="hidden md:flex flex-col justify-between p-12"
        style={{ background: "var(--bg-muted)" }}
      >
        <Logo size={40} showWordmark />
        <div>
          <p className="font-mono text-[11px] tracking-[0.16em] uppercase mb-4" style={{ color: "var(--fg-subtle)" }}>
            WELCOME BACK
          </p>
          <h2 className="font-serif text-[44px] leading-[1.05] tracking-[-0.03em] mb-4">
            색을 다시,<br />또렷하게.
          </h2>
          <p className="text-[15px] max-w-[340px] leading-relaxed" style={{ color: "var(--fg-muted)" }}>
            진단 기록과 보정 프로필을 저장해 모든 기기에서 동일한 경험을 이어가세요.
          </p>
        </div>
        <div className="font-mono text-[11px] tracking-[0.08em]" style={{ color: "var(--fg-subtle)" }}>
          v0.3 · alpha
        </div>
      </div>

      {/* Right: form */}
      <div className="flex items-center justify-center p-8 md:p-12">
        <form onSubmit={onSubmit} className="w-full max-w-[360px] flex flex-col gap-5">
          <div>
            <p className="font-mono text-[11px] tracking-[0.16em] uppercase mb-2" style={{ color: "var(--fg-subtle)" }}>
              SIGN IN
            </p>
            <h1 className="font-serif text-[32px] tracking-[-0.02em]">로그인</h1>
          </div>

          <label className="flex flex-col gap-1.5">
            <span className="font-mono text-[11px] tracking-[0.08em]" style={{ color: "var(--fg-muted)" }}>EMAIL</span>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="px-3.5 py-3 rounded-lg border text-[14px] focus:outline-none focus:ring-2 focus:ring-brand/30 transition"
              style={{ background: "var(--bg-elevated)", borderColor: "var(--border-strong)", color: "var(--fg)" }}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="font-mono text-[11px] tracking-[0.08em]" style={{ color: "var(--fg-muted)" }}>PASSWORD</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="px-3.5 py-3 rounded-lg border text-[14px] focus:outline-none focus:ring-2 focus:ring-brand/30 transition"
              style={{ background: "var(--bg-elevated)", borderColor: "var(--border-strong)", color: "var(--fg)" }}
            />
          </label>

          {error && (
            <div className="text-[13px] px-3 py-2 rounded-lg" style={{ background: "#fdecea", color: "#b23a2a" }}>
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-full bg-brand hover:bg-brand-ink text-white text-sm font-medium transition-colors disabled:opacity-60"
          >
            {loading ? "로그인 중…" : "로그인"}
          </button>

          <p className="text-center text-[13px]" style={{ color: "var(--fg-muted)" }}>
            계정이 없으신가요?{" "}
            <Link href="/register" className="text-brand hover:underline">회원가입</Link>
          </p>
        </form>
      </div>
    </div>
  );
}