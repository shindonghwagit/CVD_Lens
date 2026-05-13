"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSession, signOut } from "next-auth/react";
import Logo from "./Logo";

const NAV_ITEMS = [
  { href: "/", label: "Home" },
  { href: "/ishihara", label: "진단" },
  { href: "/correction?tab=image", label: "이미지", match: "/correction" },
  { href: "/correction", label: "실시간", match: "/correction" },
];

export default function NavBar() {
  const pathname = usePathname();
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
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.match ?? item.href.split("?")[0]);
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
        <span
          className="font-mono text-[11px] tracking-[0.04em] hidden md:flex items-center"
          style={{ color: "var(--fg-subtle)" }}
        >
          <span
            className="inline-block w-[7px] h-[7px] rounded-full mr-1.5"
            style={{
              background: "#2f9e6b",
              boxShadow: "0 0 0 3px color-mix(in oklab, #2f9e6b 20%, transparent)",
            }}
          />
          Model ready
        </span>
        {status === "authenticated" ? (
          <button
            onClick={() => signOut()}
            className="px-4 py-1.5 rounded-full text-[13px] border hover:bg-[var(--bg-muted)] transition-colors"
            style={{ borderColor: "var(--border-strong)", color: "var(--fg)" }}
          >
            로그아웃
          </button>
        ) : (
          <Link
            href="/login"
            className="px-4 py-1.5 rounded-full text-[13px] text-white bg-brand hover:bg-brand-ink transition-colors"
          >
            로그인
          </Link>
        )}
      </div>
    </header>
  );
}