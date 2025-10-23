# Phase 2: Backend Integration - Implementation Plan

## 📊 Backend Architecture Analysis

### Existing Infrastructure ✅

#### 1. **Middleware** (`/middleware/`)
- ✅ `auth.py` - JWT authentication with Supabase
  - `@require_auth` - Validates token, adds `g.user_id`, `g.user_email`, `g.access_token`
  - `@optional_auth` - Optional authentication

- ✅ `subscription.py` - Subscription validation
  - `@require_subscription` - Ensures active subscription, adds `g.subscription`, `g.plan`
  - `@check_upload_limit` - Validates upload quota
  - `@check_processing_limit` - Validates processing quota
  - `@check_storage_limit` - Validates storage quota

#### 2. **Usage Tracking** (`/utils/usage_tracking.py`)
- ✅ `increment_upload_count(user_id)` - Increments uploads
- ✅ `increment_processing_count(user_id)` - Increments processing
- ✅ `update_storage_usage(user_id, storage_mb)` - Updates storage
- ✅ `get_usage_stats(user_id)` - Gets current usage

#### 3. **Training Routes** (`/api/routes/training.py`)
**Key Routes for Quota Integration:**

| Route | Purpose | Quota Type | Current State |
|-------|---------|------------|---------------|
| `/upload-chunk` | CSV file upload | Processing | ❌ No quota check |
| `/csv-files` POST | Create CSV metadata | Processing | ❌ No quota check |
| `/generate-datasets/<session_id>` | Dataset generation | Processing | ❌ No quota check |
| `/train-models/<session_id>` | Model training | Training | ❌ No quota check |

---

## 🎯 Phase 2 Goals

### Goal 1: Add Training Quota Support
- Add `check_training_limit` decorator to middleware
- Add `increment_training_count` to usage tracking

### Goal 2: Integrate Middleware into Training Routes
- Protect CSV upload routes
- Protect dataset generation route
- Protect model training route

### Goal 3: Add Usage Tracking Calls
- Track processing jobs when CSV uploaded
- Track training runs when model training starts

---

## 📝 Implementation Steps

### Step 1: Update Middleware (`middleware/subscription.py`)

**Add training quota decorator:**

```python
def check_training_limit(f):
    """
    Decorator to check training limit before starting training

    Must be used AFTER @require_auth and @require_subscription decorators

    Usage:
        @require_auth
        @require_subscription
        @check_training_limit
        def train_models_route():
            # Training will only proceed if user has permission and quota
            return jsonify({'message': 'Training started'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'user_id') or not hasattr(g, 'plan'):
            logger.error("check_training_limit used without require_auth and require_subscription")
            return jsonify({'error': 'Authentication and subscription required'}), 401

        # Check if training is allowed for this plan
        can_use_training = g.plan.get('can_use_training', False)

        if not can_use_training:
            logger.warning(f"Training not available for user {g.user_email}'s plan")
            return jsonify({
                'error': 'Training not available',
                'message': 'Training is not available in your plan. Upgrade to Pro or Enterprise to unlock model training.',
                'plan': g.plan.get('name')
            }), 403

        # Get current usage
        usage = get_user_usage(g.user_id, g.access_token)
        training_used = usage.get('training_runs_count', 0)
        training_limit = g.plan.get('max_training_runs_per_month', 0)

        # Check if limit is unlimited (-1)
        if training_limit == -1:
            logger.info(f"Unlimited training for {g.user_email}")
            g.usage = usage
            return f(*args, **kwargs)

        # Check limit
        if training_used >= training_limit:
            logger.warning(f"Training limit reached for user {g.user_email}: {training_used}/{training_limit}")
            return jsonify({
                'error': 'Training limit reached',
                'message': f'You have reached your monthly training limit of {training_limit} runs',
                'current_usage': training_used,
                'limit': training_limit,
                'plan': g.plan.get('name')
            }), 403

        # Add usage info to g object
        g.usage = usage
        g.training_remaining = training_limit - training_used

        logger.info(f"Training check passed for {g.user_email}: {training_used}/{training_limit} used")

        return f(*args, **kwargs)

    return decorated_function
```

---

### Step 2: Update Usage Tracking (`utils/usage_tracking.py`)

**Add training count tracking:**

```python
def increment_training_count(user_id: str) -> bool:
    """
    Increment training runs count for user in current period

    Args:
        user_id: User ID

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        period_start = get_current_period_start()

        # Try to get existing record
        response = supabase.table('usage_tracking') \
            .select('*') \
            .eq('user_id', user_id) \
            .eq('period_start', period_start.isoformat()) \
            .execute()

        if response.data and len(response.data) > 0:
            # Update existing record
            usage_id = response.data[0]['id']
            current_count = response.data[0].get('training_runs_count', 0)

            supabase.table('usage_tracking') \
                .update({'training_runs_count': current_count + 1}) \
                .eq('id', usage_id) \
                .execute()

            logger.info(f"Incremented training count for user {user_id}: {current_count} -> {current_count + 1}")
        else:
            # Create new record
            supabase.table('usage_tracking') \
                .insert({
                    'user_id': user_id,
                    'period_start': period_start.isoformat(),
                    'uploads_count': 0,
                    'processing_jobs_count': 0,
                    'training_runs_count': 1,
                    'storage_used_gb': 0.0
                }) \
                .execute()

            logger.info(f"Created new usage tracking record for user {user_id} with training count: 1")

        return True

    except Exception as e:
        logger.error(f"Error incrementing training count: {str(e)}")
        return False
```

---

### Step 3: Integrate Middleware into Training Routes

#### 3.1 CSV Upload Route (`/upload-chunk`)

**Before:**
```python
@bp.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    # ... existing code ...
```

**After:**
```python
from middleware.auth import require_auth
from middleware.subscription import require_subscription, check_processing_limit
from utils.usage_tracking import increment_processing_count

@bp.route('/upload-chunk', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def upload_chunk():
    try:
        # ... existing upload logic ...

        # After successful file assembly
        if file_fully_assembled:
            # Track processing job
            increment_processing_count(g.user_id)
            logger.info(f"Tracked processing job for user {g.user_id}")

        # ... rest of code ...
    except Exception as e:
        # ... error handling ...
```

#### 3.2 CSV Files Create Route (`/csv-files` POST)

**Before:**
```python
@bp.route('/csv-files', methods=['POST'])
def create_csv_file():
    # ... existing code ...
```

**After:**
```python
@bp.route('/csv-files', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def create_csv_file():
    try:
        # ... existing logic ...

        # After successful CSV file creation
        increment_processing_count(g.user_id)
        logger.info(f"Tracked processing job for user {g.user_id}")

        # ... rest of code ...
    except Exception as e:
        # ... error handling ...
```

#### 3.3 Dataset Generation Route (`/generate-datasets/<session_id>`)

**Before:**
```python
@bp.route('/generate-datasets/<session_id>', methods=['POST'])
def generate_datasets(session_id):
    # ... existing code ...
```

**After:**
```python
@bp.route('/generate-datasets/<session_id>', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def generate_datasets(session_id):
    try:
        # ... existing logic ...

        # After successful dataset generation start
        increment_processing_count(g.user_id)
        logger.info(f"Tracked dataset generation for user {g.user_id}")

        # ... rest of code ...
    except Exception as e:
        # ... error handling ...
```

#### 3.4 Model Training Route (`/train-models/<session_id>`)

**Before:**
```python
@bp.route('/train-models/<session_id>', methods=['POST'])
def train_models(session_id):
    # ... existing code ...
```

**After:**
```python
@bp.route('/train-models/<session_id>', methods=['POST'])
@require_auth
@require_subscription
@check_training_limit
def train_models(session_id):
    try:
        # ... existing logic ...

        # After successful training start
        increment_training_count(g.user_id)
        logger.info(f"Tracked training run for user {g.user_id}")

        # ... rest of code ...
    except Exception as e:
        # ... error handling ...
```

---

### Step 4: Database Schema Updates

**Required SQL Migration:**

```sql
-- Add max_training_runs_per_month to subscription_plans table
ALTER TABLE subscription_plans
ADD COLUMN IF NOT EXISTS max_training_runs_per_month INTEGER DEFAULT 0;

-- Set default values for existing plans
UPDATE subscription_plans
SET max_training_runs_per_month = CASE
  WHEN name = 'Free' THEN 0
  WHEN name = 'Pro' THEN 5
  WHEN name = 'Enterprise' THEN -1  -- unlimited
  ELSE 0
END
WHERE max_training_runs_per_month IS NULL OR max_training_runs_per_month = 0;

-- Verify
SELECT name, can_use_training, max_training_runs_per_month
FROM subscription_plans;
```

**Update `get_user_usage` in middleware:**

```python
def get_user_usage(user_id: str, access_token: str) -> dict:
    """Get current usage for user in current billing period"""
    try:
        supabase = get_supabase_user_client(access_token)

        # Call the personalized period start function
        period_response = supabase.rpc('get_current_period_start').execute()
        period_start = period_response.data

        # Get usage tracking for current period
        response = supabase.table('usage_tracking') \
            .select('*') \
            .eq('user_id', user_id) \
            .gte('period_start', period_start) \
            .maybe_single() \
            .execute()

        if response and response.data:
            return response.data

        # No usage record yet for this period, return zeros
        logger.info(f"No usage record for user {user_id} in current period")
        return {
            'uploads_count': 0,
            'processing_jobs_count': 0,
            'training_runs_count': 0,
            'storage_used_gb': 0.0
        }

    except Exception as e:
        logger.error(f"Error fetching usage: {str(e)}")
        return {
            'uploads_count': 0,
            'processing_jobs_count': 0,
            'training_runs_count': 0,
            'storage_used_gb': 0.0
        }
```

---

## 🧪 Testing Plan

### Test 1: Quota Enforcement at Backend

```bash
# Get user token
TOKEN="your-jwt-token"

# Test 1: Upload CSV without quota
curl -X POST http://localhost:8080/api/training/upload-chunk \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{...}'

# Expected: 403 Forbidden if quota exceeded
# Response: {"error": "Processing limit reached", "message": "..."}

# Test 2: Train model without permission
curl -X POST http://localhost:8080/api/training/train-models/session-id \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{...}'

# Expected: 403 Forbidden if can_use_training = false
# Response: {"error": "Training not available", "message": "..."}
```

### Test 2: Usage Tracking

```sql
-- Before operation
SELECT processing_jobs_count, training_runs_count
FROM usage_tracking
WHERE user_id = 'user-uuid';

-- After CSV upload
-- Should see processing_jobs_count + 1

-- After training start
-- Should see training_runs_count + 1
```

---

## 📊 Success Criteria

### ✅ Phase 2 Complete When:

1. **Middleware Added** ✅
   - `check_training_limit` decorator exists
   - `increment_training_count` function exists

2. **Routes Protected** ✅
   - `/upload-chunk` has quota middleware
   - `/csv-files` POST has quota middleware
   - `/generate-datasets` has quota middleware
   - `/train-models` has quota + training middleware

3. **Usage Tracked** ✅
   - CSV upload increments `processing_jobs_count`
   - Training start increments `training_runs_count`

4. **Database Updated** ✅
   - `max_training_runs_per_month` column exists
   - Plans have correct limits set

5. **Tests Pass** ✅
   - Backend blocks operations when quota exceeded
   - Usage counts increment correctly
   - Error messages are clear and helpful

---

## 🚀 Implementation Order

### Priority 1 (Critical):
1. Add `check_training_limit` to middleware
2. Add `increment_training_count` to usage tracking
3. Add database column `max_training_runs_per_month`

### Priority 2 (High):
4. Protect `/train-models` route
5. Protect `/generate-datasets` route
6. Add usage tracking calls

### Priority 3 (Medium):
7. Protect `/upload-chunk` route
8. Protect `/csv-files` route
9. Test all routes

---

## 📁 Files to Modify

1. ✅ `/middleware/subscription.py` - Add `check_training_limit`
2. ✅ `/utils/usage_tracking.py` - Add `increment_training_count`
3. ✅ `/api/routes/training.py` - Add decorators to routes
4. ✅ Database - Run SQL migration

---

## ⚠️ Important Notes

### Column Name Inconsistencies:
The database uses different naming conventions:
- Frontend expects: `processing_jobs_count`, `training_runs_count`, `storage_used_gb`
- Database may have: `processing_count`, `storage_used_mb`

**Action:** Verify actual column names in usage_tracking table before proceeding.

### Personalized Billing Periods:
Users have 30-day billing cycles starting from registration date, NOT calendar month.

**Action:** Use `get_current_period_start()` RPC function for accurate period calculation.

### Error Handling:
All middleware must provide clear, actionable error messages to users.

**Format:**
```json
{
  "error": "Short error type",
  "message": "User-friendly explanation with upgrade path",
  "current_usage": 10,
  "limit": 10,
  "plan": "Free"
}
```
