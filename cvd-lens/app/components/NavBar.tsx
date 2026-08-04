"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useSession, signOut } from "next-auth/react";
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
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
  const [menuOpen, setMenuOpen] = useState(false);

  const isItemActive = (item: (typeof NAV_ITEMS)[number]) => {
    if (item.href === "/") return pathname === "/";
    if ("matchTab" in item) return pathname === item.matchPath && tab === item.matchTab;
    return pathname === item.href || pathname.startsWith(item.href + "/");
  };

  const close = () => setMenuOpen(false);

  // Lock body scroll while the drawer is open; restore on close/unmount (no leak).
  useEffect(() => {
    if (!menuOpen) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, [menuOpen]);

  // Rendered via a portal to <body>: the header sets backdrop-filter, which makes
  // it the containing block for position:fixed descendants — a fixed drawer nested
  // in the header would size to the header strip, not the viewport. Portalling to
  // body escapes that so inset-0 covers the whole screen.
  const drawer = (
    <div className="md:hidden">
      {/* Dimmed backdrop — tap to close */}
      <div className="fixed inset-0 z-[60] bg-black/40" onClick={close} aria-hidden="true" />
      {/* Opaque side panel */}
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="메뉴"
        className="fixed top-0 right-0 z-[61] h-full w-72 max-w-[85%] flex flex-col shadow-2xl"
        style={{ background: "var(--bg)" }}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b" style={{ borderColor: "var(--border)" }}>
          <Link href="/" className="flex items-center" onClick={close}>
            <Logo size={28} showWordmark />
          </Link>
          <button
            type="button"
            onClick={close}
            aria-label="메뉴 닫기"
            className="flex items-center justify-center w-11 h-11 -mr-2 rounded-full transition-colors"
            style={{ color: "var(--fg)" }}
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="6" y1="6" x2="18" y2="18" />
              <line x1="18" y1="6" x2="6" y2="18" />
            </svg>
          </button>
        </div>

        <nav className="flex flex-col gap-1 p-3 overflow-y-auto">
          {NAV_ITEMS.map((item) => {
            const isActive = isItemActive(item);
            return (
              <Link
                key={item.label}
                href={item.href}
                onClick={close}
                className="flex items-center min-h-[44px] px-4 rounded-xl text-[15px] whitespace-nowrap transition-colors"
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

        <div className="mt-auto p-3 border-t flex flex-col gap-2" style={{ borderColor: "var(--border)" }}>
          {status === "authenticated" ? (
            <>
              {session?.user?.name && (
                <span className="px-2 font-mono text-[12px]" style={{ color: "var(--fg-muted)" }}>
                  {session.user.name}
                </span>
              )}
              <button
                onClick={() => { close(); signOut(); }}
                className="w-full min-h-[44px] px-4 rounded-full text-[14px] whitespace-nowrap border hover:bg-[var(--bg-muted)] transition-colors"
                style={{ borderColor: "var(--border-strong)", color: "var(--fg)" }}
              >
                로그아웃
              </button>
            </>
          ) : (
            <>
              <Link
                href="/register"
                onClick={close}
                className="w-full min-h-[44px] flex items-center justify-center px-4 rounded-full text-[14px] whitespace-nowrap border transition-colors hover:bg-[var(--bg-muted)]"
                style={{ borderColor: "var(--border-strong)", color: "var(--fg)" }}
              >
                회원가입
              </Link>
              <Link
                href="/login"
                onClick={close}
                className="w-full min-h-[44px] flex items-center justify-center px-4 rounded-full text-[14px] whitespace-nowrap text-white bg-brand hover:bg-brand-ink transition-colors"
              >
                로그인
              </Link>
            </>
          )}
        </div>
      </aside>
    </div>
  );

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

      {/* Desktop nav (md+): unchanged, items forced onto one line */}
      <nav className="hidden md:flex items-center gap-1">
        {NAV_ITEMS.map((item) => {
          const isActive = isItemActive(item);
          return (
            <Link
              key={item.label}
              href={item.href}
              className="px-3.5 py-2 rounded-full text-[13px] whitespace-nowrap transition-colors"
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

      {/* Desktop session (md+) */}
      <div className="hidden md:flex items-center gap-3">
        {status === "authenticated" && session?.user?.name && (
          <span className="font-mono text-[12px]" style={{ color: "var(--fg-muted)" }}>
            {session.user.name}
          </span>
        )}
        {status === "authenticated" ? (
          <button
            onClick={() => signOut()}
            className="px-4 py-1.5 rounded-full text-[13px] whitespace-nowrap border hover:bg-[var(--bg-muted)] transition-colors"
            style={{ borderColor: "var(--border-strong)", color: "var(--fg)" }}
          >
            로그아웃
          </button>
        ) : (
          <div className="flex items-center gap-2">
            <Link
              href="/register"
              className="px-4 py-1.5 rounded-full text-[13px] whitespace-nowrap border transition-colors hover:bg-[var(--bg-muted)]"
              style={{ borderColor: "var(--border-strong)", color: "var(--fg)" }}
            >
              회원가입
            </Link>
            <Link
              href="/login"
              className="px-4 py-1.5 rounded-full text-[13px] whitespace-nowrap text-white bg-brand hover:bg-brand-ink transition-colors"
            >
              로그인
            </Link>
          </div>
        )}
      </div>

      {/* Mobile hamburger (below md): only logo + this button stay pinned */}
      <button
        type="button"
        onClick={() => setMenuOpen(true)}
        aria-label="메뉴 열기"
        aria-expanded={menuOpen}
        className="md:hidden flex items-center justify-center w-11 h-11 -mr-2 rounded-full transition-colors"
        style={{ color: "var(--fg)" }}
      >
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>

      {/* Mobile drawer — portalled to <body> to escape the header's backdrop-filter
          containing block (otherwise fixed inset-0 sizes to the header strip). */}
      {menuOpen && typeof document !== "undefined" && createPortal(drawer, document.body)}
    </header>
  );
}
