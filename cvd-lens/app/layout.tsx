import type { Metadata } from "next";
import { Suspense } from "react";
import "./globals.css";
import NavBar from "./components/NavBar";
import { ModelProvider } from "./context/ModelContext";
import SessionWrapper from "./components/SessionWrapper";

export const metadata: Metadata = {
  title: "CVDLens | 색각이상 보정 시스템",
  description: "AI 기반 색각이상 보정 웹 애플리케이션",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko" data-theme="light" className="h-full antialiased">
      <body className="min-h-full flex flex-col">
        <SessionWrapper>
          <ModelProvider>
            <Suspense fallback={null}>
              <NavBar />
            </Suspense>
            <main className="flex-1">{children}</main>
          </ModelProvider>
        </SessionWrapper>
      </body>
    </html>
  );
}
