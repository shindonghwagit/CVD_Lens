import { NextRequest, NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import pool from "@/lib/db";
import { randomUUID } from "crypto";
import { readJsonObject } from "@/lib/requestValidation";

export async function POST(req: NextRequest) {
  const body = await readJsonObject(req, 8_192);
  const email = typeof body?.email === "string" ? body.email.trim().toLowerCase() : "";
  const password = typeof body?.password === "string" ? body.password : "";
  const name = typeof body?.name === "string" ? body.name.trim() : "";

  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) || email.length > 254) {
    return NextResponse.json({ error: "올바른 이메일을 입력해주세요." }, { status: 400 });
  }

  if (password.length < 12 || password.length > 128) {
    return NextResponse.json({ error: "비밀번호는 12~128자여야 합니다." }, { status: 400 });
  }
  if (!/^[가-힣A-Za-z0-9_]{2,30}$/.test(name)) {
    return NextResponse.json({ error: "아이디는 한글, 영문, 숫자, 밑줄 2~30자로 입력해주세요." }, { status: 400 });
  }

  const { rows: emailRows } = await pool.query("SELECT id FROM users WHERE email = $1", [email]);
  if (emailRows.length > 0) {
    return NextResponse.json({ error: "이미 사용 중인 이메일입니다." }, { status: 409 });
  }

  const { rows: nameRows } = await pool.query("SELECT id FROM users WHERE name = $1", [name]);
  if (nameRows.length > 0) {
    return NextResponse.json({ error: "이미 사용 중인 아이디입니다." }, { status: 409 });
  }

  const hashed = await bcrypt.hash(password, 10);
  try {
    await pool.query(
      "INSERT INTO users (id, email, password, name) VALUES ($1, $2, $3, $4)",
      [randomUUID(), email, hashed, name]
    );
  } catch (error: unknown) {
    if ((error as { code?: string }).code === "23505") {
      return NextResponse.json({ error: "이미 사용 중인 이메일 또는 아이디입니다." }, { status: 409 });
    }
    throw error;
  }

  return NextResponse.json({ ok: true });
}
