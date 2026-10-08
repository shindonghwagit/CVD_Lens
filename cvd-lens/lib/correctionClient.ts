export const MAX_UPLOAD_SIDE = 2048;
export const REQUEST_TIMEOUT_MS = 30_000;
export const HISTORY_THUMBNAIL_SIZE = 256;

export function imageDataToURL(imageData: ImageData): string {
  const canvas = document.createElement("canvas");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d")!.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.92);
}

export function resizeDataURL(src: string, size: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;
      canvas.getContext("2d")!.drawImage(image, 0, 0, size, size);
      resolve(canvas.toDataURL("image/jpeg", 0.75));
    };
    image.onerror = () => reject(new Error("이미지를 불러오지 못했습니다."));
    image.src = src;
  });
}
