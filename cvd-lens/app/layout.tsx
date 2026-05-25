import type { Metadata } from "next";
import { Inter, Noto_Sans_KR, Instrument_Serif, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import NavBar from "./components/NavBar";
import { ModelProvider } from "./context/ModelContext";
import SessionWrapper from "./components/SessionWrapper";
import { Suspense } from "react";

const inter = Inter({ variable: "--font-sans", subsets: ["latin"] });
const notoKR = Noto_Sans_KR({ variable: "--font-kr", subsets: ["latin"], weight: ["400", "500", "700"] });
const serif = Instrument_Serif({ variable: "--font-serif", subsets: ["latin"], weight: "400" });
const mono = JetBrains_Mono({ variable: "--font-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "CVDLens · 색각이상 보정 시스템",
  description: "AI 색각이상 보정",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko" data-theme="light" className={`${inter.variable} ${notoKR.variable} ${serif.variable} ${mono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <SessionWrapper>
          <ModelProvider>
            <Suspense><NavBar /></Suspense>
            <main className="flex-1">{children}</main>
          </ModelProvider>
        </SessionWrapper>
      </body>
    </html>
  );
}