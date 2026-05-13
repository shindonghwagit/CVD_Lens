import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div style={{
        width: "100%", height: "100%",
        background: "#2a5fd9",
        borderRadius: 8,
        display: "flex", alignItems: "center", justifyContent: "center",
        position: "relative",
      }}>
        <div style={{ width: 14, height: 14, borderRadius: "50%", background: "#fff" }} />
        <div style={{ position: "absolute", width: 6, height: 6, borderRadius: "50%", background: "#183c8f" }} />
      </div>
    ),
    { ...size }
  );
}