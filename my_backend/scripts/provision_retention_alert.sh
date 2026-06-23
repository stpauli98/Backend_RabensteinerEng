#!/usr/bin/env bash
# Idempotent: creates a Cloud Monitoring log-absence alert for the retention sweep.
set -euo pipefail
PROJECT=entropia-460611
EMAIL="${1:?usage: provision_retention_alert.sh <alert-email>}"

# 1) Log-based metric counting successful sweep results.
gcloud logging metrics describe retention_sweep_ran --project "$PROJECT" >/dev/null 2>&1 || \
gcloud logging metrics create retention_sweep_ran --project "$PROJECT" \
  --description="Retention sweep completed runs" \
  --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="entropia" AND textPayload:"Retention sweep result"'

# 2) Notification channel (email).
CH=$(gcloud beta monitoring channels list --project "$PROJECT" \
      --filter="type='email' AND labels.email_address='$EMAIL'" --format='value(name)' | head -1)
if [ -z "$CH" ]; then
  CH=$(gcloud beta monitoring channels create --project "$PROJECT" \
        --display-name="Retention alerts" --type=email \
        --channel-labels="email_address=$EMAIL" --format='value(name)')
fi

# 3) Alert policy: fire when the sweep metric is ABSENT for 24h (Cloud Monitoring
#    caps absence duration at 1 day; the sweep logs a result on every boot-tick,
#    so a full day with no result is genuinely anomalous). Created from a JSON
#    policy file for gcloud-version stability.
if ! gcloud alpha monitoring policies list --project "$PROJECT" \
      --filter="displayName='Retention sweep stale'" --format='value(name)' | grep -q .; then
  POLICY_FILE=$(mktemp)
  cat > "$POLICY_FILE" <<JSON
{
  "displayName": "Retention sweep stale",
  "combiner": "OR",
  "conditions": [{
    "displayName": "No sweep result in 23h30m",
    "conditionAbsent": {
      "filter": "metric.type=\"logging.googleapis.com/user/retention_sweep_ran\" AND resource.type=\"cloud_run_revision\"",
      "duration": "84600s",
      "aggregations": [{"alignmentPeriod": "3600s", "perSeriesAligner": "ALIGN_COUNT"}]
    }
  }],
  "notificationChannels": ["$CH"],
  "enabled": true
}
JSON
  gcloud alpha monitoring policies create --project "$PROJECT" --policy-from-file="$POLICY_FILE"
  rm -f "$POLICY_FILE"
fi
echo "Provisioned retention staleness alert."
