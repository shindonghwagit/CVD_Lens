"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useSession, signOut } from "next-auth/react";
import Logo from "./Logo";

const NAV_ITEMS = [
  { href: "/",                    label: "Home" },
  { href: "/ishihara",            label: "진단" },
  { href: "/history",             label: "진단 기록" },
  { href: "/corrections",         label: "보정 기록" },
  { href: "/correction?tab=image",label: "이미지",  matchPath: "/correction", matchTab: "image" },
  { href: "/correction",          label: "카메라",  matchPath: "/correction", matchTab: "" },
];

export default function NavBar() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const tab = searchParams.get("tab") ?? "";
  const { data: session, status } = useSession();

  return (
    <header
      className="sticky top-0 z-40 flex items-center justify-between px-6 md:px-10 py-4 border-b"
      style={{
        background: "color-mix(in oklab, var(--bg) 82%, transparent)",
        backdropFilter: "blur(16px)",
        borderColor: "var(--border)",
      }}
    >
      {/* Brand */}
      <Link href="/" className="flex items-center">
        <Logo size={32} showWordmark />
      </Link>

      {/* Nav */}
      <nav className="flex items-center gap-1">
        {NAV_ITEMS.map((item) => {
          let isActive: boolean;
          if (item.href === "/") {
            isActive = pathname === "/";
          } else if ("matchTab" in item) {
            isActive = pathname === item.matchPath && tab === item.matchTab;
          } else {
            isActive = pathname === item.href || pathname.startsWith(item.href + "/");
          }
          return (
            <Link
              key={item.label}
              href={item.href}
              className="px-3.5 py-2 rounded-full text-[13px] transition-colors"
              style={{
                background: isActive ? "var(--bg-elevated)" : "transparent",
                color: isActive ? "var(--fg)" : "var(--fg-muted)",
                boxShadow: isActive ? "var(--shadow-soft)" : "none",
              }}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Session */}
      <div className="flex items-center gap-3">
        {status === "authenticated" && session?.user?.name && (
          <span className="font-mono text-[12px] hidden md:block" style={{ color: "var(--fg-muted)" }}>
            {session.user.name}
          </span>
        )}
        {status === "authenticated" ? (
          <button
            onClick={() => signOut()}
            className="px-4 py-1.5 rounded-full text-[13px] border hover:bg-[var(--bg-muted)] transition-colors"
            style={{ borderColor: "var(--border-strong)", color: "var(--fg)" }}
          >
            로그아웃
          </button>
        ) : (
          <div className="flex items-center gap-2">
            <Link
              href="/register"
              className="px-4 py-1.5 rounded-full text-[13px] border transition-colors hover:bg-[var(--bg-muted)]"
              style={{ borderColor: "var(--border-strong)", color: "var(--fg)" }}
            >
              회원가입
            </Link>
            <Link
              href="/login"
              className="px-4 py-1.5 rounded-full text-[13px] text-white bg-brand hover:bg-brand-ink transition-colors"
            >
              로그인
            </Link>
          </div>
        )}
      </div>
    </header>
  );
}
