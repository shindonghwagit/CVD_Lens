CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY,
  email text NOT NULL UNIQUE,
  password text NOT NULL,
  name text NOT NULL UNIQUE,
  preferred_cvd_type varchar(1) CHECK (preferred_cvd_type IN ('p', 'd', 't')),
  created_at timestamptz NOT NULL DEFAULT now(),
  last_login_at timestamptz
);

CREATE TABLE IF NOT EXISTS ishihara_results (
  id uuid PRIMARY KEY,
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  correct integer NOT NULL CHECK (correct >= 0),
  total integer NOT NULL CHECK (total BETWEEN 1 AND 38 AND correct <= total),
  diagnosis text NOT NULL CHECK (diagnosis IN ('normal', 'protan', 'deutan', 'rg')),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS correction_results (
  id uuid PRIMARY KEY,
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  cvd_type varchar(1) NOT NULL CHECK (cvd_type IN ('p', 'd', 't')),
  source text NOT NULL DEFAULT 'image' CHECK (source IN ('image', 'camera', 'video')),
  original_image text NOT NULL,
  corrected_image text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ishihara_results_user_created_idx
  ON ishihara_results (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS correction_results_user_created_idx
  ON correction_results (user_id, created_at DESC);
