# Free User Quota Testing Guide

**Date**: 2025-10-24
**Test User**: test@rabensteiner.com
**Status**: ✅ CONFIGURED

## User Configuration

### User Details
- **Email**: test@rabensteiner.com
- **User ID**: f4e69951-af93-4db8-9521-eadc4021e13c
- **Current Plan**: Free (active)
- **Previous Plan**: Pro (cancelled)

### Free Plan Limits
```json
{
  "plan_name": "Free",
  "can_use_training": false,
  "max_training_runs_per_month": 0,
  "max_processing_jobs_per_month": 5,
  "max_uploads_per_month": 5,
  "max_file_size_mb": 10,
  "max_storage_gb": 0.5
}
```

### Current Usage (Period: 2025-10-21 to 2025-11-19)
```json
{
  "uploads_count": 0,
  "processing_jobs_count": 0,
  "training_runs_count": 0,
  "storage_used_gb": 0.00,
  "operations_count": 0
}
```

## How to Get JWT Token

### Method 1: Login through Frontend
1. Go to your application login page
2. Login with:
   - Email: test@rabensteiner.com
   - Password: [your test password]
3. Open browser DevTools → Application → Local Storage
4. Find the JWT token (usually stored as `supabase.auth.token` or similar)

### Method 2: Using Supabase Client
```bash
# Using curl to login
curl -X POST 'https://luvjebsltuttakatnzaa.supabase.co/auth/v1/token?grant_type=password' \
  -H "apikey: YOUR_SUPABASE_ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@rabensteiner.com",
    "password": "YOUR_TEST_PASSWORD"
  }'

# Response will contain access_token
```

## Test Scenarios

Once you have the JWT token, save it to environment variable:
```bash
export FREE_USER_TOKEN="eyJhbGciOiJIUzI1NiIs..."
```

### Test 1: Free User Cannot Train (CRITICAL)
```bash
echo "=== Test 1: Free user attempts training (expect 403 - Training not available) ==="
curl -s -X POST http://localhost:8081/api/training/train-models/test123 \
  -H "Authorization: Bearer ${FREE_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"model_parameters": {}}' | jq .

# Expected Response:
# {
#   "error": "Training not available",
#   "message": "Training is not available in your plan. Upgrade to Pro or Enterprise to unlock model training.",
#   "plan": "Free"
# }
```

### Test 2: Free User Cannot Generate Datasets (CRITICAL)
```bash
echo "=== Test 2: Free user attempts dataset generation (expect 403 if at limit) ==="
curl -s -X POST http://localhost:8081/api/training/generate-datasets/test123 \
  -H "Authorization: Bearer ${FREE_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"model_parameters": {}}' | jq .

# Expected Response (if 0/5 used):
# {
#   "success": true,
#   "message": "Datasets generated successfully"
# }

# Expected Response (if 5/5 used - at limit):
# {
#   "error": "Processing limit reached",
#   "message": "You have reached your monthly processing limit of 5",
#   "current_usage": 5,
#   "limit": 5,
#   "plan": "Free"
# }
```

### Test 3: Free User Processing Limit
```bash
# Generate datasets 5 times to reach limit
for i in {1..5}; do
  echo "=== Attempt $i/5 ==="
  curl -s -X POST http://localhost:8081/api/training/generate-datasets/test_session_$i \
    -H "Authorization: Bearer ${FREE_USER_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"model_parameters": {}}' | jq .
  sleep 2
done

# 6th attempt should fail
echo "=== Attempt 6/5 (should fail) ==="
curl -s -X POST http://localhost:8081/api/training/generate-datasets/test_session_6 \
  -H "Authorization: Bearer ${FREE_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"model_parameters": {}}' | jq .

# Expected: {"error": "Processing limit reached", "current_usage": 5, "limit": 5}
```

### Test 4: Check Current Usage
```bash
echo "=== Check usage stats ==="
curl -s http://localhost:8081/api/subscription/usage \
  -H "Authorization: Bearer ${FREE_USER_TOKEN}" | jq .

# Expected Response:
# {
#   "uploads_count": 0,
#   "processing_jobs_count": X,  # Number of generate-datasets calls
#   "training_runs_count": 0,    # Should remain 0 (training blocked)
#   "storage_used_gb": 0.00
# }
```

## Validation Checklist

### ✅ Free Plan Restrictions
- [ ] Cannot execute train-models endpoint (403 - "Training not available")
- [ ] Cannot exceed 5 processing operations per month
- [ ] Cannot exceed 5 uploads per month
- [ ] Cannot upload files larger than 10MB
- [ ] Cannot exceed 0.5GB storage

### ✅ Middleware Chain
- [ ] @require_auth validates JWT token
- [ ] @require_subscription validates active subscription
- [ ] @check_training_limit blocks training for Free users
- [ ] @check_processing_limit blocks when limit reached

## Reset Usage for Testing

If you need to reset usage counts for testing:

```sql
-- Reset usage for current period
UPDATE usage_tracking 
SET 
  uploads_count = 0,
  processing_jobs_count = 0,
  training_runs_count = 0,
  storage_used_gb = 0.00,
  operations_count = 0,
  updated_at = NOW()
WHERE user_id = 'f4e69951-af93-4db8-9521-eadc4021e13c'
  AND period_start = '2025-10-21';
```

## Expected Behavior Summary

| Action | Free User | Expected Result |
|--------|-----------|----------------|
| Login | ✅ | Success |
| Upload files (1-5) | ✅ | Success until limit |
| Upload files (6+) | ❌ | 403 - Upload limit reached |
| Generate datasets (1-5) | ✅ | Success until limit |
| Generate datasets (6+) | ❌ | 403 - Processing limit reached |
| Train models | ❌ | 403 - Training not available |

## Upgrade Path Testing

After testing Free user limits, you can test upgrade by:

1. Upgrade test@rabensteiner.com to Pro plan
2. Verify training becomes available
3. Verify higher limits (200 processing/month, 5 trainings/month)

```sql
-- Upgrade to Pro (for testing)
UPDATE user_subscriptions 
SET plan_id = '651ef3a7-e619-423e-b312-394a410ad2e2',  -- Pro plan ID
    status = 'active',
    updated_at = NOW()
WHERE user_id = 'f4e69951-af93-4db8-9521-eadc4021e13c'
  AND status = 'active';
```

