"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { useModel } from "../context/ModelContext";
import { CVDType } from "../hooks/useCVDModel";

const CVD_LABELS: Record<CVDType, string> = {
  p: "Protanopia (제1색맹)",
  d: "Deuteranopia (제2색맹)",
  t: "Tritanopia (제3색맹)",
};

const MODEL_SIZE = 512; // 업로드 이미지는 고해상도 처리

export default function ImageCorrection() {
  const { ready, error, infer } = useModel();
  const [cvdType, setCvdType] = useState<CVDType>("d");
  const [original, setOriginal] = useState<string | null>(null);
  const [corrected, setCorrected] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [sliderX, setSliderX] = useState(50); // 0~100 %
  const [dragging, setDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 이미지 처리
  const processImage = useCallback(async (file: File, type: CVDType) => {
    if (!ready) return;
    setProcessing(true);
    setCorrected(null);

    const bitmap = await createImageBitmap(file);
    const canvas = document.createElement("canvas");
    canvas.width = MODEL_SIZE;
    canvas.height = MODEL_SIZE;
    const ctx = canvas.getContext("2d")!;

    // 중앙 크롭 → 256×256
    const side = Math.min(bitmap.width, bitmap.height);
    const sx = (bitmap.width - side) / 2;
    const sy = (bitmap.height - side) / 2;
    ctx.drawImage(bitmap, sx, sy, side, side, 0, 0, MODEL_SIZE, MODEL_SIZE);

    setOriginal(canvas.toDataURL());

    const imageData = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
    const result = await infer(imageData, type);

    ctx.putImageData(result, 0, 0);
    setCorrected(canvas.toDataURL());
    setProcessing(false);
  }, [ready, infer]);

  // 파일 선택
  const onFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    processImage(file, cvdType);
  }, [processImage, cvdType]);

  const pendingFileRef = useRef<File | null>(null);
  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) { pendingFileRef.current = file; onFile(file); }
  };

  // CVD 타입 변경 시 재처리 (파일이 있을 때만, ready 변화는 무시)
  const prevCvdType = useRef(cvdType);
  useEffect(() => {
    if (prevCvdType.current === cvdType) return;
    prevCvdType.current = cvdType;
    if (pendingFileRef.current && ready) {
      processImage(pendingFileRef.current, cvdType);
    }
  }, [cvdType, processImage, ready]);

  // 드래그 앤 드롭
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) { pendingFileRef.current = file; onFile(file); }
  };

  // 슬라이더 드래그
  const onSliderMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    setSliderX(x);
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent | TouchEvent) => {
      const x = "touches" in e ? e.touches[0].clientX : e.clientX;
      onSliderMove(x);
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
              cvdType === type ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-300 hover:bg-gray-700"
            }`}
          >
            {CVD_LABELS[type]}
          </button>
        ))}
      </div>

      {/* 업로드 영역 */}
      {!original && (
        <div
          onDrop={onDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => fileInputRef.current?.click()}
          className="w-full max-w-sm h-48 border-2 border-dashed border-gray-600 rounded-xl flex flex-col items-center justify-center gap-2 cursor-pointer hover:border-blue-500 hover:bg-gray-900 transition-colors"
        >
          <svg className="w-10 h-10 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <p className="text-gray-400 text-sm">이미지를 드래그하거나 클릭해서 업로드</p>
          <p className="text-gray-600 text-xs">JPG, PNG 지원</p>
          <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={onFileChange} />
        </div>
      )}

      {/* 비교 슬라이더 */}
      {original && (
        <div className="flex flex-col items-center gap-4 w-full max-w-sm">
          {/* 슬라이더 뷰 */}
          <div
            ref={containerRef}
            className="relative w-full max-w-lg aspect-square rounded-xl overflow-hidden cursor-ew-resize select-none border border-gray-700"
            onMouseDown={() => setDragging(true)}
            onTouchStart={() => setDragging(true)}
          >
            {/* 보정 이미지 (뒤) */}
            {corrected && (
              <img src={corrected} alt="corrected" className="absolute inset-0 w-full h-full object-cover" draggable={false} />
            )}
            {/* 원본 이미지 (앞, clip) */}
            {original && (
              <div
                className="absolute inset-0 overflow-hidden"
                style={{ width: `${sliderX}%` }}
              >
                <img src={original} alt="original" className="absolute inset-0 w-full h-full max-w-none object-cover" draggable={false} />
              </div>
            )}
            {/* 슬라이더 선 */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg"
              style={{ left: `${sliderX}%` }}
            >
              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-white shadow-lg flex items-center justify-center">
                <svg className="w-4 h-4 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l-4 3 4 3M16 9l4 3-4 3" />
                </svg>
              </div>
            </div>
            {/* 레이블 */}
            <span className="absolute top-2 left-2 text-xs bg-black/50 text-white px-2 py-0.5 rounded-full">원본</span>
            <span className="absolute top-2 right-2 text-xs bg-blue-600/80 text-white px-2 py-0.5 rounded-full">보정</span>

            {processing && (
              <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                <p className="text-white text-sm">처리 중...</p>
              </div>
            )}
          </div>

          {/* 다른 이미지 버튼 */}
          <button
            onClick={() => { setOriginal(null); setCorrected(null); pendingFileRef.current = null; fileInputRef.current && (fileInputRef.current.value = ""); }}
            className="text-sm text-gray-400 hover:text-white transition-colors"
          >
            다른 이미지 선택
          </button>
        </div>
      )}

      {!ready && <p className="text-yellow-400 text-sm">모델 로딩 중...</p>}
      {error && <p className="text-red-400 text-sm">오류: {error}</p>}
    </div>
  );
}
