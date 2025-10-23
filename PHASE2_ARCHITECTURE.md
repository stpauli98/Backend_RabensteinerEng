# Phase 2: Backend Architecture Overview

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                         │
│  - useTrainingQuota hook validates quota                        │
│  - Shows error messages if quota exceeded                       │
│  - Blocks UI operations                                         │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP Request with JWT
                     │ Authorization: Bearer <token>
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (Flask/Python)                      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              MIDDLEWARE CHAIN                           │    │
│  │                                                          │    │
│  │  1️⃣ @require_auth                                       │    │
│  │     ├─ Validates JWT token                             │    │
│  │     ├─ Extracts user_id, user_email                    │    │
│  │     └─ Stores in g.user_id, g.access_token             │    │
│  │                                                          │    │
│  │  2️⃣ @require_subscription                               │    │
│  │     ├─ Fetches active subscription from DB             │    │
│  │     ├─ Gets plan details (limits, features)            │    │
│  │     └─ Stores in g.subscription, g.plan                │    │
│  │                                                          │    │
│  │  3️⃣ @check_processing_limit (for CSV/datasets)         │    │
│  │     ├─ Gets current usage from usage_tracking          │    │
│  │     ├─ Compares with plan.max_processing_jobs          │    │
│  │     ├─ Returns 403 if limit exceeded                   │    │
│  │     └─ Stores in g.usage, g.processing_remaining       │    │
│  │                                                          │    │
│  │  4️⃣ @check_training_limit (for model training)         │    │
│  │     ├─ Checks plan.can_use_training = true             │    │
│  │     ├─ Gets current training_runs_count                │    │
│  │     ├─ Compares with plan.max_training_runs            │    │
│  │     ├─ Returns 403 if no permission or limit exceeded  │    │
│  │     └─ Stores in g.training_remaining                  │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                   ROUTE HANDLERS                        │    │
│  │                                                          │    │
│  │  📤 /upload-chunk                                       │    │
│  │     @require_auth                                       │    │
│  │     @require_subscription                               │    │
│  │     @check_processing_limit                             │    │
│  │     ├─ Validates CSV file                              │    │
│  │     ├─ Saves to storage                                │    │
│  │     └─ increment_processing_count(user_id)  ← TRACK    │    │
│  │                                                          │    │
│  │  📝 /csv-files [POST]                                   │    │
│  │     @require_auth                                       │    │
│  │     @require_subscription                               │    │
│  │     @check_processing_limit                             │    │
│  │     ├─ Creates CSV metadata                            │    │
│  │     └─ increment_processing_count(user_id)  ← TRACK    │    │
│  │                                                          │    │
│  │  🎲 /generate-datasets/<session_id>                     │    │
│  │     @require_auth                                       │    │
│  │     @require_subscription                               │    │
│  │     @check_processing_limit                             │    │
│  │     ├─ Starts dataset generation                       │    │
│  │     └─ increment_processing_count(user_id)  ← TRACK    │    │
│  │                                                          │    │
│  │  🤖 /train-models/<session_id>                          │    │
│  │     @require_auth                                       │    │
│  │     @require_subscription                               │    │
│  │     @check_training_limit                               │    │
│  │     ├─ Starts model training                           │    │
│  │     └─ increment_training_count(user_id)  ← TRACK      │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              USAGE TRACKING UTILITIES                   │    │
│  │                                                          │    │
│  │  increment_processing_count(user_id)                    │    │
│  │     ├─ Gets current period start                       │    │
│  │     ├─ Finds usage_tracking record                     │    │
│  │     └─ Increments processing_jobs_count                │    │
│  │                                                          │    │
│  │  increment_training_count(user_id)                      │    │
│  │     ├─ Gets current period start                       │    │
│  │     ├─ Finds usage_tracking record                     │    │
│  │     └─ Increments training_runs_count                  │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────────────────┘
                     │ Supabase Client Queries
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SUPABASE DATABASE                           │
│                                                                  │
│  📊 subscription_plans                                           │
│     ├─ name, price_monthly, price_yearly                        │
│     ├─ max_uploads_per_month                                    │
│     ├─ max_processing_jobs_per_month                            │
│     ├─ max_storage_gb                                           │
│     ├─ can_use_training                                         │
│     └─ max_training_runs_per_month  ← NEW COLUMN                │
│                                                                  │
│  👤 user_subscriptions                                           │
│     ├─ user_id                                                  │
│     ├─ plan_id                                                  │
│     ├─ status (active/cancelled)                                │
│     ├─ started_at, expires_at                                   │
│     └─ period_start, period_end                                 │
│                                                                  │
│  📈 usage_tracking                                               │
│     ├─ user_id                                                  │
│     ├─ period_start, period_end                                 │
│     ├─ operations_count                                         │
│     ├─ uploads_count                                            │
│     ├─ processing_jobs_count  ← TRACKED HERE                    │
│     ├─ training_runs_count    ← TRACKED HERE                    │
│     └─ storage_used_gb                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Request Flow Example

### Scenario: User Uploads CSV File

```
1. Frontend
   ├─ User selects CSV file
   ├─ useTrainingQuota checks: canProcessDataset = true ✅
   ├─ Sends POST /upload-chunk with JWT token
   └─ { chunk_data, session_id, filename }

2. Backend Middleware Chain
   ├─ @require_auth
   │  ├─ Validates JWT token
   │  ├─ g.user_id = "abc-123"
   │  └─ ✅ Continue
   │
   ├─ @require_subscription
   │  ├─ Queries: SELECT * FROM user_subscriptions WHERE user_id = 'abc-123'
   │  ├─ g.subscription = {...}
   │  ├─ g.plan = {name: "Pro", max_processing_jobs_per_month: 400}
   │  └─ ✅ Continue
   │
   └─ @check_processing_limit
      ├─ Queries: SELECT processing_jobs_count FROM usage_tracking
      ├─ current: 350, limit: 400
      ├─ 350 < 400 ✅
      └─ ✅ Continue

3. Route Handler: upload_chunk()
   ├─ Validates chunk data
   ├─ Assembles file from chunks
   ├─ Saves to Supabase Storage
   ├─ 📈 increment_processing_count(user_id)  ← USAGE TRACKED
   │  └─ UPDATE usage_tracking SET processing_jobs_count = 351
   └─ Return success response

4. Frontend
   └─ Shows success message
```

### Scenario: User Tries Training Without Permission

```
1. Frontend
   ├─ User clicks "Train Model"
   ├─ useTrainingQuota checks: canTrainModel = false ❌
   ├─ Shows error: "Training is not available in your plan..."
   └─ ❌ Request NOT sent (blocked by frontend)

BUT if someone bypasses frontend and calls API directly:

2. Backend Middleware Chain
   ├─ @require_auth → ✅ Valid token
   ├─ @require_subscription → ✅ Active subscription
   └─ @check_training_limit
      ├─ Checks: plan.can_use_training = false ❌
      └─ ❌ Returns 403 Forbidden
         {
           "error": "Training not available",
           "message": "Training is not available in your plan. Upgrade to Pro..."
         }

3. Request blocked at backend ✅ (Defense in depth)
```

---

## 🔐 Security: Defense in Depth

```
Layer 1: Frontend Validation
├─ useTrainingQuota hook checks quota
├─ Disables buttons if quota exceeded
└─ Shows user-friendly error messages
   └─ ⚠️ Can be bypassed by direct API calls

Layer 2: Backend Middleware (CRITICAL)
├─ @require_auth validates JWT
├─ @require_subscription validates active subscription
├─ @check_*_limit validates quota limits
└─ ✅ CANNOT be bypassed
   └─ 🛡️ True enforcement layer

Layer 3: Database Constraints
├─ RLS policies restrict data access
├─ Foreign keys ensure referential integrity
└─ ✅ Final safety net
```

**Principle:** Always enforce security at backend, frontend is just UX optimization.

---

## 📊 Database Schema Updates

### Current State:
```sql
CREATE TABLE subscription_plans (
  id UUID PRIMARY KEY,
  name VARCHAR(50),
  max_processing_jobs_per_month INTEGER,
  can_use_training BOOLEAN,
  -- ❌ max_training_runs_per_month NOT EXISTS
);
```

### Required Migration:
```sql
ALTER TABLE subscription_plans
ADD COLUMN max_training_runs_per_month INTEGER DEFAULT 0;

UPDATE subscription_plans
SET max_training_runs_per_month = CASE
  WHEN name = 'Free' THEN 0
  WHEN name = 'Pro' THEN 5
  WHEN name = 'Enterprise' THEN -1  -- unlimited
END;
```

### After Migration:
```sql
CREATE TABLE subscription_plans (
  id UUID PRIMARY KEY,
  name VARCHAR(50),
  max_processing_jobs_per_month INTEGER,
  can_use_training BOOLEAN,
  max_training_runs_per_month INTEGER, -- ✅ ADDED
);
```

---

## 🧪 Testing Strategy

### Unit Tests (Python):
```python
def test_check_training_limit_no_permission():
    """User with Free plan cannot train models"""
    g.plan = {'name': 'Free', 'can_use_training': False}
    response = check_training_limit(mock_route)()
    assert response[1] == 403
    assert 'Training not available' in response[0].json['error']

def test_check_training_limit_quota_exceeded():
    """User with Pro plan but quota exceeded"""
    g.plan = {'name': 'Pro', 'can_use_training': True, 'max_training_runs_per_month': 5}
    g.usage = {'training_runs_count': 5}
    response = check_training_limit(mock_route)()
    assert response[1] == 403
    assert 'Training limit reached' in response[0].json['error']
```

### Integration Tests (curl):
```bash
# Test 1: Upload CSV without auth
curl -X POST http://localhost:8080/api/training/upload-chunk
# Expected: 401 Unauthorized

# Test 2: Upload CSV with Free plan (10/10 quota)
curl -X POST http://localhost:8080/api/training/upload-chunk \
  -H "Authorization: Bearer $TOKEN"
# Expected: 403 Forbidden (quota exceeded)

# Test 3: Train model with Free plan
curl -X POST http://localhost:8080/api/training/train-models/session-id \
  -H "Authorization: Bearer $TOKEN"
# Expected: 403 Forbidden (training not available)
```

---

## 📈 Metrics & Monitoring

### Key Metrics to Track:
1. **Quota Rejections** - Count of 403 responses by reason
2. **Usage Trends** - Processing/training count over time
3. **Plan Upgrades** - Conversions after hitting limits
4. **Error Rates** - Authentication vs quota vs other errors

### Logging:
```python
logger.info(f"Processing check passed: {used}/{limit} for {user_email}")
logger.warning(f"Processing limit reached: {used}/{limit} for {user_email}")
logger.error(f"Authentication failed: {error}")
```

---

## ✅ Phase 2 Completion Checklist

- [ ] `check_training_limit` decorator created
- [ ] `increment_training_count` function created
- [ ] Database column `max_training_runs_per_month` added
- [ ] Plans updated with training limits
- [ ] `/upload-chunk` route protected
- [ ] `/csv-files` POST route protected
- [ ] `/generate-datasets` route protected
- [ ] `/train-models` route protected
- [ ] Usage tracking calls added to all routes
- [ ] Unit tests written for middleware
- [ ] Integration tests with curl passed
- [ ] Error messages are user-friendly
- [ ] Logging added for all quota checks
- [ ] Documentation updated

**Ready to implement? Let's start! 🚀**
