import { NextRequest } from "next/server";

export async function readJsonObject(req: NextRequest, maxBytes = 400_000): Promise<Record<string, unknown> | null> {
  const declared = Number(req.headers.get("content-length") ?? 0);
  if (declared > maxBytes) return null;
  const text = await req.text();
  if (!text || new TextEncoder().encode(text).byteLength > maxBytes) return null;
  try {
    const value: unknown = JSON.parse(text);
    return value !== null && typeof value === "object" && !Array.isArray(value)
      ? value as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

export function isDataImage(value: unknown): value is string {
  return typeof value === "string"
    && value.length <= 300_000
    && /^data:image\/jpeg;base64,[A-Za-z0-9+/]+={0,2}$/.test(value);
}
