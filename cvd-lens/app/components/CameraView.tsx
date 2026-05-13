"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CVDType, useCVDModel } from "../hooks/useCVDModel";

const CVD_LABELS: Record<CVDType, string> = {
  p: "Protanopia (제1색맹)",
  d: "Deuteranopia (제2색맹)",
  t: "Tritanopia (제3색맹)",
};

export default function CameraView() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [cvdType, setCvdType] = useState<CVDType>("d");
  const [streaming, setStreaming] = useState(false);
  const [original, setOriginal] = useState<string | null>(null);
  const [corrected, setCorrected] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [sliderX, setSliderX] = useState(50);
  const [dragging, setDragging] = useState(false);

  const { ready, error, infer } = useCVDModel();

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });
      videoRef.current!.srcObject = stream;
      await videoRef.current!.play();
      setStreaming(true);
    } catch {
      alert("카메라 접근에 실패했습니다.");
    }
  }

  function stopCamera() {
    const video = videoRef.current;
    if (video?.srcObject) {
      (video.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    setStreaming(false);
  }

  const capture = useCallback(async () => {
    const video = videoRef.current;
    if (!video || !ready) return;

    setProcessing(true);
    setCorrected(null);

    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d")!;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const side = Math.min(vw, vh);
    ctx.drawImage(video, (vw - side) / 2, (vh - side) / 2, side, side, 0, 0, 512, 512);

    setOriginal(canvas.toDataURL("image/jpeg", 0.9));
    stopCamera();

    try {
      const imageData = ctx.getImageData(0, 0, 512, 512);
      const result = await infer(imageData, cvdType);
      ctx.putImageData(result, 0, 0);
      setCorrected(canvas.toDataURL("image/jpeg", 0.9));
    } catch (e) {
      console.error("보정 오류:", e);
    }
    setProcessing(false);
  }, [ready, cvdType, infer]);

  // 슬라이더 드래그
  const onSliderMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    setSliderX(Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100)));
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent | TouchEvent) => {
      onSliderMove("touches" in e ? e.touches[0].clientX : e.clientX);
    };
    const onUp = () => setDragging(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchmove", onMove);
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onUp);
    };
  }, [dragging, onSliderMove]);

  return (
    <div className="flex flex-col items-center gap-6">
      {/* CVD 타입 선택 */}
      <div className="flex gap-2 flex-wrap justify-center">
        {(Object.keys(CVD_LABELS) as CVDType[]).map((type) => (
          <button
            key={type}
            onClick={() => setCvdType(type)}
            className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
              cvdType === type
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-300 hover:bg-gray-700"
            }`}
          >
            {CVD_LABELS[type]}
          </button>
        ))}
      </div>

      {!original ? (
        /* 카메라 프리뷰 */
        <div className="flex flex-col items-center gap-4 w-full">
          <div className="relative w-full max-w-sm aspect-square rounded-xl overflow-hidden border border-gray-700 bg-black">
            <video ref={videoRef} className="w-full h-full object-cover" playsInline muted />
            {!streaming && (
              <div className="absolute inset-0 flex items-center justify-center">
                <p className="text-gray-500 text-sm">카메라가 꺼져 있습니다</p>
              </div>
            )}
          </div>
          <div className="flex gap-3">
            {!streaming ? (
              <button
                onClick={startCamera}
                disabled={!ready}
                className="px-6 py-2.5 rounded-full bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-40 transition-colors"
              >
                카메라 시작
              </button>
            ) : (
              <>
                <button
                  onClick={capture}
                  className="px-6 py-2.5 rounded-full bg-white text-gray-900 text-sm font-semibold hover:bg-gray-100 transition-colors"
                >
                  촬영
                </button>
                <button
                  onClick={stopCamera}
                  className="px-6 py-2.5 rounded-full bg-gray-800 text-gray-300 text-sm font-medium hover:bg-gray-700 transition-colors"
                >
                  취소
                </button>
              </>
            )}
          </div>
        </div>
      ) : (
        /* 보정 결과 슬라이더 */
        <div className="flex flex-col items-center gap-4 w-full max-w-sm">
          <div
            ref={containerRef}
            className="relative w-full aspect-square rounded-xl overflow-hidden cursor-ew-resize select-none border border-gray-700"
            onMouseDown={() => setDragging(true)}
            onTouchStart={() => setDragging(true)}
          >
            {corrected && (
              <img src={corrected} alt="corrected" className="absolute inset-0 w-full h-full object-cover" draggable={false} />
            )}
            <div className="absolute inset-0 overflow-hidden" style={{ width: `${sliderX}%` }}>
              <img src={original} alt="original" className="absolute inset-0 w-full h-full max-w-none object-cover" draggable={false} />
            </div>
            <div className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg" style={{ left: `${sliderX}%` }}>
              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-white shadow-lg flex items-center justify-center">
                <svg className="w-4 h-4 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l-4 3 4 3M16 9l4 3-4 3" />
                </svg>
              </div>
            </div>
            <span className="absolute top-2 left-2 text-xs bg-black/50 text-white px-2 py-0.5 rounded-full">원본</span>
            <span className="absolute top-2 right-2 text-xs bg-blue-600/80 text-white px-2 py-0.5 rounded-full">보정</span>
            {processing && (
              <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                <p className="text-white text-sm">처리 중...</p>
              </div>
            )}
          </div>
          <button
            onClick={() => { setOriginal(null); setCorrected(null); }}
            className="text-sm text-gray-400 hover:text-white transition-colors"
          >
            다시 촬영
          </button>
        </div>
      )}

      {!ready && <p className="text-yellow-400 text-sm">서버 연결 중...</p>}
      {error && <p className="text-red-400 text-sm">오류: {error}</p>}
    </div>
  );
}
