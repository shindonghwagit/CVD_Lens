import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import pool from "@/lib/db";
import { randomUUID } from "crypto";
import { isDataImage, readJsonObject } from "@/lib/requestValidation";

export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const body = await readJsonObject(req);
  const cvdType = body?.cvdType;
  const source = body?.source ?? "image";
  const originalImage = body?.originalImage;
  const correctedImage = body?.correctedImage;
  if (!(["p", "d", "t"] as unknown[]).includes(cvdType)
    || !(["image", "camera", "video"] as unknown[]).includes(source)
    || !isDataImage(originalImage) || !isDataImage(correctedImage)) {
    return NextResponse.json({ error: "잘못된 보정 결과입니다." }, { status: 400 });
  }

  const { rows } = await pool.query(
    `INSERT INTO correction_results (id, user_id, cvd_type, source, original_image, corrected_image)
     VALUES ($1, $2, $3, $4, $5, $6) RETURNING *`,
    [randomUUID(), session.user.id, cvdType, source, originalImage, correctedImage]
  );

  return NextResponse.json(rows[0]);
}

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const { rows } = await pool.query(
    "SELECT id, cvd_type, source, original_image, corrected_image, created_at FROM correction_results WHERE user_id = $1 ORDER BY created_at DESC LIMIT 30",
    [session.user.id]
  );

  return NextResponse.json(rows);
}

export async function DELETE(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const id = req.nextUrl.searchParams.get("id");
  if (!id || !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(id)) {
    return NextResponse.json({ error: "잘못된 기록 ID입니다." }, { status: 400 });
  }

  const { rowCount } = await pool.query(
    "DELETE FROM correction_results WHERE id = $1 AND user_id = $2",
    [id, session.user.id]
  );

  if (!rowCount) {
    return NextResponse.json({ error: "기록을 찾을 수 없습니다." }, { status: 404 });
  }
  return NextResponse.json({ ok: true });
}
