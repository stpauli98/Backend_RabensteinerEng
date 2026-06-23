-- Retention notification & deletion hardening (2026-06-23)

ALTER TABLE user_subscriptions
  ADD COLUMN IF NOT EXISTS scheduled_deletion_at timestamptz;

CREATE TABLE IF NOT EXISTS retention_notices (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subscription_id   uuid NOT NULL REFERENCES user_subscriptions(id) ON DELETE CASCADE,
  user_id           uuid NOT NULL,
  kind              text NOT NULL CHECK (kind IN ('warn1','warn2')),
  resend_message_id text,
  status            text NOT NULL DEFAULT 'sending'
                    CHECK (status IN ('sending','sent','delivered','bounced','complained','failed')),
  sent_at           timestamptz,
  created_at        timestamptz NOT NULL DEFAULT now(),
  updated_at        timestamptz NOT NULL DEFAULT now(),
  UNIQUE (subscription_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_retention_notices_message_id
  ON retention_notices (resend_message_id);
CREATE INDEX IF NOT EXISTS idx_retention_notices_user_id
  ON retention_notices (user_id);

-- Backfill existing warn timestamps into notices (status 'sent', no message_id).
INSERT INTO retention_notices (subscription_id, user_id, kind, status, sent_at, created_at)
SELECT id, user_id, 'warn1', 'sent', retention_warn1_sent_at, retention_warn1_sent_at
FROM user_subscriptions
WHERE retention_warn1_sent_at IS NOT NULL
ON CONFLICT (subscription_id, kind) DO NOTHING;

INSERT INTO retention_notices (subscription_id, user_id, kind, status, sent_at, created_at)
SELECT id, user_id, 'warn2', 'sent', retention_warn2_sent_at, retention_warn2_sent_at
FROM user_subscriptions
WHERE retention_warn2_sent_at IS NOT NULL
ON CONFLICT (subscription_id, kind) DO NOTHING;

-- Backfill scheduled_deletion_at for already-warned rows: max(expires_at + 30d, warn1 + 7d).
UPDATE user_subscriptions
SET scheduled_deletion_at = GREATEST(expires_at + interval '30 days',
                                     retention_warn1_sent_at + interval '7 days')
WHERE retention_warn1_sent_at IS NOT NULL
  AND scheduled_deletion_at IS NULL;
