import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Suspense } from "react";
import "./globals.css";
import NavBar from "./components/NavBar";
import { ModelProvider } from "./context/ModelContext";
import SessionWrapper from "./components/SessionWrapper";

const inter = Inter({ subsets: ["latin"], variable: "--f-sans", display: "swap" });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--f-mono", display: "swap" });

export const metadata: Metadata = {
  title: "CVDLens | 색각이상 보정 시스템",
  description: "AI 기반 색각이상 보정 웹 애플리케이션",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko" data-theme="light" className={`h-full antialiased ${inter.variable} ${jetbrainsMono.variable}`}>
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
