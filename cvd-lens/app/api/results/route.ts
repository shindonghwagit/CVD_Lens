import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import pool from "@/lib/db";
import { randomUUID } from "crypto";

export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const { correct, total, diagnosis } = await req.json();

  const { rows } = await pool.query(
    "INSERT INTO ishihara_results (id, user_id, correct, total, diagnosis) VALUES ($1, $2, $3, $4, $5) RETURNING *",
    [randomUUID(), session.user.id, correct, total, diagnosis]
  );

  return NextResponse.json(rows[0]);
}

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const { rows } = await pool.query(
    "SELECT * FROM ishihara_results WHERE user_id = $1 ORDER BY created_at DESC LIMIT 10",
    [session.user.id]
  );

  return NextResponse.json(rows);
}
