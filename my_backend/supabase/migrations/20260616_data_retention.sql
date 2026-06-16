-- Data retention after subscription expiry (Trello #101)

ALTER TABLE public.user_subscriptions
  ADD COLUMN IF NOT EXISTS retention_warn1_sent_at timestamptz,
  ADD COLUMN IF NOT EXISTS retention_warn2_sent_at timestamptz,
  ADD COLUMN IF NOT EXISTS data_deleted_at         timestamptz;

-- Single-row daily-claim lock so only one Cloud Run instance runs the sweep/day.
CREATE TABLE IF NOT EXISTS public.retention_sweep_runs (
  id              integer PRIMARY KEY DEFAULT 1,
  last_started_at timestamptz,
  CONSTRAINT retention_sweep_runs_singleton CHECK (id = 1)
);
INSERT INTO public.retention_sweep_runs (id, last_started_at)
  VALUES (1, NULL)
  ON CONFLICT (id) DO NOTHING;
