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
      --filter="type=email AND labels.email_address=$EMAIL" --format='value(name)' | head -1)
if [ -z "$CH" ]; then
  CH=$(gcloud beta monitoring channels create --project "$PROJECT" \
        --display-name="Retention alerts" --type=email \
        --channel-labels="email_address=$EMAIL" --format='value(name)')
fi

# 3) Alert policy: metric absent for 36h.
gcloud alpha monitoring policies list --project "$PROJECT" \
  --filter='displayName="Retention sweep stale"' --format='value(name)' | grep -q . || \
gcloud alpha monitoring policies create --project "$PROJECT" \
  --display-name="Retention sweep stale" \
  --notification-channels="$CH" \
  --condition-display-name="No sweep result in 36h" \
  --condition-filter='metric.type="logging.googleapis.com/user/retention_sweep_ran" AND resource.type="cloud_run_revision"' \
  --condition-threshold-comparison=COMPARISON_LT \
  --condition-threshold-value=1 \
  --condition-threshold-duration=129600s \
  --aggregation='{"alignmentPeriod":"3600s","perSeriesAligner":"ALIGN_COUNT"}'
echo "Provisioned retention staleness alert."
