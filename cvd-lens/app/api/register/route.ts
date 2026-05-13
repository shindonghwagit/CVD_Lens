import { NextRequest, NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import pool from "@/lib/db";
import { randomUUID } from "crypto";

export async function POST(req: NextRequest) {
  const { email, password, name } = await req.json();

  if (!email || !password || !name) {
    return NextResponse.json({ error: "모든 필드를 입력해주세요." }, { status: 400 });
  }

  const { rows } = await pool.query("SELECT id FROM users WHERE email = $1", [email]);
  if (rows.length > 0) {
    return NextResponse.json({ error: "이미 사용 중인 이메일입니다." }, { status: 409 });
  }

  const hashed = await bcrypt.hash(password, 10);
  await pool.query(
    "INSERT INTO users (id, email, password, name) VALUES ($1, $2, $3, $4)",
    [randomUUID(), email, hashed, name]
  );

  return NextResponse.json({ ok: true });
}
