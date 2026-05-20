import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import pool from "@/lib/db";
import { randomUUID } from "crypto";

export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const { cvdType, source } = await req.json();

  const { rows } = await pool.query(
    "INSERT INTO correction_results (id, user_id, cvd_type, source) VALUES ($1, $2, $3, $4) RETURNING *",
    [randomUUID(), session.user.id, cvdType, source ?? "image"]
  );

  return NextResponse.json(rows[0]);
}

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const { rows } = await pool.query(
    "SELECT * FROM correction_results WHERE user_id = $1 ORDER BY created_at DESC LIMIT 20",
    [session.user.id]
  );

  return NextResponse.json(rows);
}
