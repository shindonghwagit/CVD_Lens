import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import bcrypt from "bcryptjs";
import pool from "./db";

export const { handlers, auth, signIn, signOut } = NextAuth({
  session: { strategy: "jwt" },
  secret: process.env.NEXTAUTH_SECRET ?? "cvdlens-secret-key-2026",
  pages: {
    signIn: "/login",
  },
  providers: [
    CredentialsProvider({
      name: "credentials",
      credentials: {
        username: { label: "아이디", type: "text" },
        password: { label: "비밀번호", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.username || !credentials?.password) return null;

        const { rows } = await pool.query(
          "SELECT id, email, name, password, preferred_cvd_type FROM users WHERE name = $1",
          [credentials.username]
        );
        const user = rows[0];
        if (!user) return null;

        const valid = await bcrypt.compare(credentials.password as string, user.password as string);
        if (!valid) return null;

        await pool.query("UPDATE users SET last_login_at = NOW() WHERE id = $1", [user.id]);

        return { id: user.id, email: user.email, name: user.name, preferred_cvd_type: user.preferred_cvd_type };
      },
    }),
  ],
  callbacks: {
    jwt({ token, user }) {
      if (user) token.id = user.id;
      return token;
    },
    session({ session, token }) {
      if (session.user) session.user.id = token.id as string;
      return session;
    },
  },
});
