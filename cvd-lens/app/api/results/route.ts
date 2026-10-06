import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import pool from "@/lib/db";
import { randomUUID } from "crypto";
import { readJsonObject } from "@/lib/requestValidation";

export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const body = await readJsonObject(req, 8_192);
  const correct = body?.correct;
  const total = body?.total;
  const diagnosis = body?.diagnosis;
  if (!Number.isInteger(correct) || !Number.isInteger(total)
    || (total as number) < 1 || (total as number) > 38
    || (correct as number) < 0 || (correct as number) > (total as number)
    || !(["normal", "protan", "deutan", "rg"] as unknown[]).includes(diagnosis)) {
    return NextResponse.json({ error: "잘못된 검사 결과입니다." }, { status: 400 });
  }

  const preferred = diagnosis === "protan" ? "p" : diagnosis === "normal" ? null : "d";
  const client = await pool.connect();
  let rows;
  try {
    await client.query("BEGIN");
    ({ rows } = await client.query(
      "INSERT INTO ishihara_results (id, user_id, correct, total, diagnosis) VALUES ($1, $2, $3, $4, $5) RETURNING *",
      [randomUUID(), session.user.id, correct, total, diagnosis]
    ));
    await client.query("UPDATE users SET preferred_cvd_type = $1 WHERE id = $2", [preferred, session.user.id]);
    await client.query("COMMIT");
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }

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
