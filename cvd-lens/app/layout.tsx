import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Suspense } from "react";
import "./globals.css";
import NavBar from "./components/NavBar";
import { ModelProvider } from "./context/ModelContext";
import SessionWrapper from "./components/SessionWrapper";

const inter = Inter({ subsets: ["latin"], variable: "--f-sans", display: "swap" });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--f-mono", display: "swap" });

const SITE_URL = "https://cvd-lens.vercel.app";
const OG_TITLE = "CVDLens | 색각이상 AI 보정";
const OG_DESC =
  "색각이상(적색맹·녹색맹·청색맹) 사용자를 위한 AI 색 보정 도구. 이시하라 자가 진단부터 사진·카메라 실시간 보정까지 한 곳에서.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: "CVDLens | 색각이상 보정 시스템",
  description: "AI 기반 색각이상 보정 웹 애플리케이션",
  openGraph: {
    type: "website",
    locale: "ko_KR",
    url: "/",
    siteName: "CVDLens",
    title: OG_TITLE,
    description: OG_DESC,
    images: [
      {
        url: "/landing/og_cover.jpg",
        width: 1200,
        height: 630,
        alt: "CVDLens — 붉은 버스의 원본과 녹색맹 보정본 비교",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: OG_TITLE,
    description: OG_DESC,
    images: ["/landing/og_cover.jpg"],
  },
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
