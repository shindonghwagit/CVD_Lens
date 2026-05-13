import type { Metadata } from "next";
import { Inter, Instrument_Serif, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import NavBar from "./components/NavBar";
import { ModelProvider } from "./context/ModelContext";
import SessionWrapper from "./components/SessionWrapper";

const inter = Inter({ variable: "--font-sans", subsets: ["latin"] });
const serif = Instrument_Serif({ variable: "--font-serif", subsets: ["latin"], weight: "400" });
const mono = JetBrains_Mono({ variable: "--font-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "CVDLens · 색각이상 보정 시스템",
  description: "AI 실시간 색각이상 보정",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko" data-theme="light" className={`${inter.variable} ${serif.variable} ${mono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <SessionWrapper>
          <ModelProvider>
            <NavBar />
            <main className="flex-1">{children}</main>
          </ModelProvider>
        </SessionWrapper>
      </body>
    </html>
  );
}