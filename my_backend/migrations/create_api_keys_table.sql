-- API Keys table for forecast endpoint authentication
-- Applied: 2026-04-02

CREATE TABLE api_keys (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id uuid NOT NULL,
    key_hash varchar NOT NULL UNIQUE,
    key_prefix varchar(12) NOT NULL,
    name varchar(100) NOT NULL,
    expires_at timestamptz,
    last_used_at timestamptz,
    revoked_at timestamptz,
    created_at timestamptz DEFAULT now(),
    UNIQUE(session_id, name)
);

ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can select own api_keys" ON api_keys FOR SELECT TO authenticated
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own api_keys" ON api_keys FOR INSERT TO authenticated
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own api_keys" ON api_keys FOR UPDATE TO authenticated
USING (auth.uid() = user_id);

CREATE POLICY "Service role full access api_keys" ON api_keys FOR ALL TO service_role
USING (true) WITH CHECK (true);
