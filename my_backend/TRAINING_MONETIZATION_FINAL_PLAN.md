# Training Monetization - Finalni Akcioni Plan

**Datum**: 2025-10-24
**Status**: Backend 95% ✅ | Frontend 85% ✅
**Preostalo rada**: ~14 sati

---

## 🎯 Executive Summary

### Što je već urađeno ✅

**Backend (95% Complete)**:
- ✅ JWT Authentication middleware potpuno implementiran
- ✅ Subscription service sa plan validation
- ✅ Sve 3 quota middleware funkcije (training, processing, upload)
- ✅ Usage tracking utilities za sve operacije
- ✅ Generate datasets endpoint sa quota checks
- ✅ Upload endpoints sa quota checks
- ✅ Usage increment funkcije

**Frontend (85% Complete)**:
- ✅ useTrainingQuota hook implementiran
- ✅ TrainingPage sa quota checks na train i generate
- ✅ useQuotaCheck hook za opću quota logiku
- ✅ VerificationWarningBanner komponenta
- ✅ Subscription context i hooks

### Što nedostaje ❌

**Backend (5%)**:
- ❌ 2 dekoratora na train_models endpoint (KRITIČNO)
- ⚠️ E2E testiranje quota flows

**Frontend (15%)**:
- ❌ Usage refresh nakon training-a (KRITIČNO)
- ❌ DataProcessingPage quota integracija (KRITIČNO)
- ❌ DataAdjustmentsPage quota integracija (KRITIČNO)
- ⚠️ QuotaWarningBanner komponenta
- ⚠️ QuotaCounter komponenta
- ⚠️ SubscriptionStatus widget

---

## 🚨 FAZA 1: KRITIČNE POPRAVKE (3h) - DO THIS FIRST

### B1: Dodati Dekoratore na train_models Endpoint 🔴
**Komponenta**: Backend
**Prioritet**: CRITICAL - Sigurnosna rupa u monetizaciji
**Vrijeme**: 15 min
**Lokacija**: `/my_backend/api/routes/training.py` Line 1493

**Problem**:
- Free users mogu pokrenuti training (ne bi smjeli)
- Pro users mogu preći limit od 10 training-a
- Usage se tracka ALI se ne provjerava prije izvršenja

**Rješenje**:
```python
# PRIJE:
@bp.route('/train-models/<session_id>', methods=['POST'])
@require_auth
def train_models(session_id):
    # ...

# POSLIJE:
@bp.route('/train-models/<session_id>', methods=['POST'])
@require_auth
@require_subscription
@check_training_limit
def train_models(session_id):
    # ...
```

**Verifikacija Import Statements** (Lines 1-25):
```python
from middleware.auth import require_auth
from middleware.subscription import require_subscription, check_training_limit
```

**Test Scenario**:
```bash
# Test 1: Free user - expect 403
curl -X POST http://localhost:5000/api/training/train-models/test123 \
  -H "Authorization: Bearer FREE_USER_TOKEN" \
  -d '{"model_parameters": {}}'

# Expected: {"error": "Training not available", "plan": "Free"}

# Test 2: Pro user at limit (10/10) - expect 403
curl -X POST http://localhost:5000/api/training/train-models/test123 \
  -H "Authorization: Bearer PRO_USER_AT_LIMIT_TOKEN" \
  -d '{"model_parameters": {}}'

# Expected: {"error": "Training limit reached", "current_usage": 10, "limit": 10}

# Test 3: Pro user (5/10) - expect 200
curl -X POST http://localhost:5000/api/training/train-models/test123 \
  -H "Authorization: Bearer PRO_USER_BELOW_LIMIT_TOKEN" \
  -d '{"model_parameters": {}}'

# Expected: {"success": true, "message": "Model training started"}
```

**Success Criteria**:
- ✅ Free user dobija 403 sa porukom "Training not available"
- ✅ Pro user na limitu dobija 403 sa "Training limit reached"
- ✅ Pro user ispod limita dobija 200 i training počinje
- ✅ Enterprise user UVIJEK dobija 200 (unlimited)

---

### F1: Usage Refresh Nakon Training-a 🔴
**Komponenta**: Frontend
**Prioritet**: CRITICAL - Korisnik vidi stari usage nakon training-a
**Vrijeme**: 1.5h
**Lokacija**: `/src/pages/training/TrainingPage.tsx`

**Problem**:
Kada korisnik završi training, usage count se povećava u bazi, ali frontend još uvijek prikazuje stari broj. Korisnik misli da može još jedan training, ali dobija "limit reached" grešku.

**Rješenje**:

1. **Dodati usage refresh u TrainingPage.tsx** nakon uspješnog training-a:

```typescript
// Dodati import
import { useSubscription } from '@/features/subscription/hooks';

// U komponenti
const { refreshUsage } = useSubscription();

// Modificirati handleTrainModels funkciju (Lines 452-500)
const handleTrainModels = useCallback(async () => {
  if (!canTrainModel) {
    setErrorMessage(getQuotaError('training'));
    return;
  }

  setErrorMessage(null);
  setIsTraining(true);
  setTrainingProgress({ status: 'training', progress: 0 });

  try {
    await trainingApiService.trainModels(sessionId, {
      model_parameters: modelParameters,
      training_split: trainingSplit,
    });

    // ✅ DODATI REFRESH OVDJE
    await refreshUsage(); // ← Nova linija

    setTrainingProgress({ status: 'completed', progress: 100 });
    await fetchEvaluationResults();
  } catch (error) {
    console.error('Training error:', error);
    setErrorMessage(error.message);
    setTrainingProgress({ status: 'error', progress: 0 });
  } finally {
    setIsTraining(false);
  }
}, [sessionId, modelParameters, trainingSplit, canTrainModel, getQuotaError, refreshUsage]);
```

2. **Dodati refreshUsage u useSubscription hook** ako ne postoji:

**Lokacija**: `/src/features/subscription/hooks/index.ts` (ili gdje god je hook)

```typescript
export const useSubscription = () => {
  const [subscription, setSubscription] = useState(null);
  const [usage, setUsage] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchSubscription = useCallback(async () => {
    // ... existing fetch logic
  }, []);

  const fetchUsage = useCallback(async () => {
    try {
      const response = await api.get('/api/subscription/usage');
      setUsage(response.data);
    } catch (error) {
      console.error('Failed to fetch usage:', error);
    }
  }, []);

  // ✅ DODATI OVU FUNKCIJU
  const refreshUsage = useCallback(async () => {
    await fetchUsage();
  }, [fetchUsage]);

  useEffect(() => {
    fetchSubscription();
    fetchUsage();
  }, [fetchSubscription, fetchUsage]);

  return {
    subscription,
    usage,
    loading,
    refreshUsage, // ← Export
  };
};
```

**Test Scenario**:
1. Login kao Pro user (5/10 trainings used)
2. Pogledaj QuotaCounter - prikazuje "5/10 trainings"
3. Pokreni training
4. Čekaj da se training završi
5. QuotaCounter se automatski updatea na "6/10 trainings"
6. Ne mora refresh stranice

**Success Criteria**:
- ✅ Nakon uspješnog training-a, usage count se automatski osvježava
- ✅ Korisnik vidi točan preostali broj training-a bez refresh-a
- ✅ Ako je nakon training-a limit dostignut, button "Train Models" postaje disabled

---

### F2: DataProcessingPage Quota Integracija 🔴
**Komponenta**: Frontend
**Prioritet**: CRITICAL - Processing nije zaštićen quota checks
**Vrijeme**: 45 min
**Lokacija**: `/src/pages/data-processing/DataProcessingPage.tsx`

**Problem**:
DataProcessingPage trenutno NE provjerava quota prije processing operacija. Korisnik može pokrenuti processing čak i ako je prekoračio limit.

**Rješenje**:

1. **Dodati useQuotaCheck hook**:

```typescript
import { useQuotaCheck } from '@/features/subscription/hooks/useQuotaCheck';

const DataProcessingPage = () => {
  const {
    canProcessDataset,
    getQuotaError,
    loading: quotaLoading
  } = useQuotaCheck();

  // ... existing state

  // ✅ DODATI QUOTA CHECK u process funkciju
  const handleProcess = useCallback(async () => {
    if (!canProcessDataset) {
      setErrorMessage(getQuotaError('processing'));
      return;
    }

    // ... existing processing logic
  }, [canProcessDataset, getQuotaError]);

  return (
    <div>
      {/* ✅ DODATI QUOTA WARNING */}
      {!canProcessDataset && (
        <QuotaWarningBanner
          type="processing"
          message={getQuotaError('processing')}
        />
      )}

      {/* ✅ DISABLE BUTTON ako nema quota */}
      <Button
        onClick={handleProcess}
        disabled={!canProcessDataset || quotaLoading || isProcessing}
      >
        {t('dataProcessing.startButton')}
      </Button>
    </div>
  );
};
```

2. **Dodati refresh nakon processing-a**:

```typescript
import { useSubscription } from '@/features/subscription/hooks';

const { refreshUsage } = useSubscription();

const handleProcess = useCallback(async () => {
  if (!canProcessDataset) {
    setErrorMessage(getQuotaError('processing'));
    return;
  }

  try {
    await processData(sessionId, parameters);

    // ✅ REFRESH USAGE
    await refreshUsage();

    setSuccessMessage('Processing completed successfully');
  } catch (error) {
    setErrorMessage(error.message);
  }
}, [canProcessDataset, getQuotaError, sessionId, refreshUsage]);
```

**Test Scenario**:
1. Login kao Free user (1/1 processing used)
2. Navigate to DataProcessingPage
3. Vidljiv je QuotaWarningBanner "Processing limit reached"
4. Button "Start Processing" je disabled
5. Login kao Pro user (10/20 processing available)
6. Button je enabled, processing radi
7. Nakon processing-a, usage se automatski osvježava na 11/20

**Success Criteria**:
- ✅ Free user ne može pokrenuti processing ako je limit dostignut
- ✅ Pro user vidi preostali broj processing operacija
- ✅ Nakon processing-a, usage se automatski osvježava
- ✅ Warning banner se prikazuje kad korisnik nema quota

---

### F3: DataAdjustmentsPage Quota Integracija 🔴
**Komponenta**: Frontend
**Prioritet**: CRITICAL - Adjustments nisu zaštićeni quota checks
**Vrijeme**: 45 min
**Lokacija**: `/src/pages/data-adjustments/DataAdjustmentsPage.tsx`

**Isti pattern kao F2, ali za adjustments operacije.**

**Rješenje**:

```typescript
import { useQuotaCheck } from '@/features/subscription/hooks/useQuotaCheck';
import { useSubscription } from '@/features/subscription/hooks';

const DataAdjustmentsPage = () => {
  const {
    canProcessDataset,  // Adjustments koriste processing quota
    getQuotaError,
    loading: quotaLoading
  } = useQuotaCheck();

  const { refreshUsage } = useSubscription();

  const handleAdjustData = useCallback(async () => {
    if (!canProcessDataset) {
      setErrorMessage(getQuotaError('processing'));
      return;
    }

    try {
      await adjustData(sessionId, adjustments);

      // ✅ REFRESH USAGE
      await refreshUsage();

      setSuccessMessage('Adjustments applied successfully');
    } catch (error) {
      setErrorMessage(error.message);
    }
  }, [canProcessDataset, getQuotaError, sessionId, refreshUsage]);

  return (
    <div>
      {!canProcessDataset && (
        <QuotaWarningBanner
          type="processing"
          message={getQuotaError('processing')}
        />
      )}

      <Button
        onClick={handleAdjustData}
        disabled={!canProcessDataset || quotaLoading || isAdjusting}
      >
        {t('dataAdjustments.applyButton')}
      </Button>
    </div>
  );
};
```

**Test Scenario**: Isti kao F2

**Success Criteria**: Isti kao F2

---

## ⚠️ FAZA 2: VAŽNE KOMPONENTE (6h)

### F4: QuotaWarningBanner Komponenta ⚠️
**Komponenta**: Frontend
**Prioritet**: HIGH - Koristi se u F2, F3
**Vrijeme**: 2h
**Lokacija**: `/src/shared/components/banners/QuotaWarningBanner.tsx`

**Opis**:
Reusable warning banner komponenta koja prikazuje quota upozorenja sa opcijom za upgrade.

**Implementacija**:

```typescript
import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { Button } from '@/shared/components/ui/Button';
import { useNavigate } from 'react-router-dom';

interface QuotaWarningBannerProps {
  type: 'upload' | 'processing' | 'training' | 'storage';
  message: string;
  showUpgradeButton?: boolean;
  currentUsage?: number;
  limit?: number;
}

export const QuotaWarningBanner: React.FC<QuotaWarningBannerProps> = ({
  type,
  message,
  showUpgradeButton = true,
  currentUsage,
  limit,
}) => {
  const navigate = useNavigate();

  const getBannerColor = () => {
    if (limit && currentUsage) {
      const percentage = (currentUsage / limit) * 100;
      if (percentage >= 100) return 'bg-red-100 border-red-500 text-red-800';
      if (percentage >= 80) return 'bg-orange-100 border-orange-500 text-orange-800';
      return 'bg-yellow-100 border-yellow-500 text-yellow-800';
    }
    return 'bg-red-100 border-red-500 text-red-800';
  };

  return (
    <div className={`border-l-4 p-4 mb-4 rounded ${getBannerColor()}`}>
      <div className="flex items-start">
        <AlertTriangle className="h-5 w-5 mr-3 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <h3 className="font-semibold mb-1">
            {type === 'training' && 'Training Limit Reached'}
            {type === 'processing' && 'Processing Limit Reached'}
            {type === 'upload' && 'Upload Limit Reached'}
            {type === 'storage' && 'Storage Limit Reached'}
          </h3>
          <p className="text-sm">{message}</p>
          {currentUsage !== undefined && limit !== undefined && (
            <p className="text-xs mt-2">
              Current usage: {currentUsage}/{limit === -1 ? '∞' : limit}
            </p>
          )}
        </div>
        {showUpgradeButton && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate('/subscription')}
            className="ml-4 flex-shrink-0"
          >
            Upgrade Plan
          </Button>
        )}
      </div>
    </div>
  );
};
```

**Test Scenario**:
```tsx
// Test 1: Limit reached (100%)
<QuotaWarningBanner
  type="training"
  message="You have used all 10 training runs this month"
  currentUsage={10}
  limit={10}
/>
// Expected: Red banner, "Upgrade Plan" button visible

// Test 2: Close to limit (85%)
<QuotaWarningBanner
  type="processing"
  message="You are close to your processing limit"
  currentUsage={17}
  limit={20}
/>
// Expected: Orange banner

// Test 3: No upgrade button
<QuotaWarningBanner
  type="training"
  message="Free plan does not include training"
  showUpgradeButton={false}
/>
// Expected: Red banner, no button
```

**Success Criteria**:
- ✅ Prikazuje različite boje ovisno o usage percentage (red > orange > yellow)
- ✅ "Upgrade Plan" button navigira na /subscription
- ✅ Prikazuje current usage i limit
- ✅ Radi sa svim tipovima quota (training, processing, upload, storage)

---

### F5: QuotaCounter Komponenta ⚠️
**Komponenta**: Frontend
**Prioritet**: HIGH - Real-time usage display
**Vrijeme**: 2h
**Lokacija**: `/src/shared/components/feedback/QuotaCounter.tsx`

**Opis**:
Live quota counter koji se prikazuje u svakoj stranici sa operacijama. Pokazuje trenutni usage i limit.

**Implementacija**:

```typescript
import React from 'react';
import { useSubscription } from '@/features/subscription/hooks';
import { Database, Zap, Upload, Cpu } from 'lucide-react';

type QuotaType = 'training' | 'processing' | 'upload' | 'storage';

interface QuotaCounterProps {
  type: QuotaType;
  showIcon?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export const QuotaCounter: React.FC<QuotaCounterProps> = ({
  type,
  showIcon = true,
  size = 'md',
}) => {
  const { subscription, usage, loading } = useSubscription();

  if (loading) {
    return <div className="animate-pulse bg-gray-200 h-6 w-24 rounded"></div>;
  }

  if (!subscription || !usage) {
    return null;
  }

  const getQuotaData = () => {
    switch (type) {
      case 'training':
        return {
          icon: <Zap className="h-4 w-4" />,
          label: 'Training',
          used: usage.training_runs_count || 0,
          limit: subscription.max_training_runs_per_month || 0,
        };
      case 'processing':
        return {
          icon: <Cpu className="h-4 w-4" />,
          label: 'Processing',
          used: usage.processing_count || 0,
          limit: subscription.max_processing_operations_per_month || 0,
        };
      case 'upload':
        return {
          icon: <Upload className="h-4 w-4" />,
          label: 'Uploads',
          used: usage.upload_count || 0,
          limit: subscription.max_uploads_per_month || 0,
        };
      case 'storage':
        const usedGB = (usage.storage_used_bytes || 0) / (1024 ** 3);
        const limitGB = (subscription.max_storage_bytes || 0) / (1024 ** 3);
        return {
          icon: <Database className="h-4 w-4" />,
          label: 'Storage',
          used: usedGB.toFixed(2),
          limit: limitGB === 0 ? 0 : limitGB.toFixed(2),
          unit: 'GB',
        };
      default:
        return null;
    }
  };

  const data = getQuotaData();
  if (!data) return null;

  const { icon, label, used, limit, unit = '' } = data;
  const percentage = limit === -1 ? 0 : limit === 0 ? 100 : (Number(used) / Number(limit)) * 100;

  const getColor = () => {
    if (percentage >= 100) return 'text-red-600';
    if (percentage >= 80) return 'text-orange-600';
    if (percentage >= 60) return 'text-yellow-600';
    return 'text-green-600';
  };

  const sizeClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  };

  return (
    <div className={`flex items-center space-x-2 ${sizeClasses[size]}`}>
      {showIcon && <span className={getColor()}>{icon}</span>}
      <span className="font-medium">{label}:</span>
      <span className={`font-semibold ${getColor()}`}>
        {used}/{limit === -1 ? '∞' : limit} {unit}
      </span>
      {percentage > 0 && (
        <div className="w-16 bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all ${
              percentage >= 100 ? 'bg-red-600' :
              percentage >= 80 ? 'bg-orange-600' :
              percentage >= 60 ? 'bg-yellow-600' :
              'bg-green-600'
            }`}
            style={{ width: `${Math.min(percentage, 100)}%` }}
          ></div>
        </div>
      )}
    </div>
  );
};
```

**Korištenje u TrainingPage**:

```typescript
import { QuotaCounter } from '@/shared/components/feedback/QuotaCounter';

<div className="mb-4 space-y-2">
  <QuotaCounter type="training" />
  <QuotaCounter type="processing" />
</div>
```

**Test Scenario**:
```tsx
// Test 1: Normal usage (50%)
// User: Pro plan, 5/10 trainings used
<QuotaCounter type="training" />
// Expected: Green color, "5/10", progress bar at 50%

// Test 2: High usage (90%)
// User: Pro plan, 9/10 trainings used
<QuotaCounter type="training" />
// Expected: Orange color, "9/10", progress bar at 90%

// Test 3: Limit reached (100%)
// User: Pro plan, 10/10 trainings used
<QuotaCounter type="training" />
// Expected: Red color, "10/10", progress bar at 100%

// Test 4: Unlimited
// User: Enterprise plan, unlimited trainings
<QuotaCounter type="training" />
// Expected: "25/∞", no progress bar
```

**Success Criteria**:
- ✅ Prikazuje real-time usage za sve quota tipove
- ✅ Boja se mijenja ovisno o usage percentage (green/yellow/orange/red)
- ✅ Progress bar vizualno predstavlja usage
- ✅ Podržava unlimited quota (∞)
- ✅ Podržava različite veličine (sm, md, lg)

---

### F6: SubscriptionStatus Widget ⚠️
**Komponenta**: Frontend
**Prioritet**: MEDIUM - Nice to have za NavBar
**Vrijeme**: 2h
**Lokacija**: `/src/shared/components/layout/SubscriptionStatus.tsx`

**Opis**:
Widget u NavBar-u koji prikazuje trenutni plan i omogućava brz pristup subscription postavkama.

**Implementacija**:

```typescript
import React, { useState } from 'react';
import { useSubscription } from '@/features/subscription/hooks';
import { Crown, ChevronDown } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export const SubscriptionStatus: React.FC = () => {
  const { subscription, loading } = useSubscription();
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();

  if (loading) {
    return <div className="animate-pulse bg-gray-200 h-8 w-32 rounded"></div>;
  }

  if (!subscription) {
    return (
      <button
        onClick={() => navigate('/subscription')}
        className="px-3 py-1.5 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700"
      >
        Get Started
      </button>
    );
  }

  const planName = subscription.subscription_plans?.name || 'Free';
  const planColors = {
    Free: 'bg-gray-100 text-gray-800',
    Basic: 'bg-blue-100 text-blue-800',
    Pro: 'bg-purple-100 text-purple-800',
    Enterprise: 'bg-amber-100 text-amber-800',
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center space-x-2 px-3 py-1.5 rounded-md text-sm font-medium ${
          planColors[planName] || planColors.Free
        }`}
      >
        <Crown className="h-4 w-4" />
        <span>{planName}</span>
        <ChevronDown className="h-3 w-3" />
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          ></div>
          <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-lg border z-20">
            <div className="p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="font-semibold">{planName} Plan</span>
                <Crown className="h-5 w-5 text-amber-500" />
              </div>

              <div className="space-y-2 text-sm text-gray-600 mb-4">
                <div className="flex justify-between">
                  <span>Training:</span>
                  <span className="font-medium">
                    {subscription.max_training_runs_per_month === -1
                      ? 'Unlimited'
                      : `${subscription.max_training_runs_per_month}/month`}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Processing:</span>
                  <span className="font-medium">
                    {subscription.max_processing_operations_per_month === -1
                      ? 'Unlimited'
                      : `${subscription.max_processing_operations_per_month}/month`}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Storage:</span>
                  <span className="font-medium">
                    {(subscription.max_storage_bytes / (1024 ** 3)).toFixed(0)} GB
                  </span>
                </div>
              </div>

              <button
                onClick={() => {
                  navigate('/subscription');
                  setIsOpen(false);
                }}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700"
              >
                {planName === 'Free' || planName === 'Basic' ? 'Upgrade Plan' : 'Manage Subscription'}
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
```

**Integracija u NavBar**:

```typescript
// /src/shared/components/layout/NavBar.tsx

import { SubscriptionStatus } from './SubscriptionStatus';

<nav className="bg-white shadow">
  <div className="max-w-7xl mx-auto px-4">
    <div className="flex justify-between items-center h-16">
      <div className="flex items-center">
        {/* Logo */}
      </div>

      <div className="flex items-center space-x-4">
        {/* Ostali elementi */}
        <SubscriptionStatus />
        <UserMenu />
      </div>
    </div>
  </div>
</nav>
```

**Test Scenario**:
1. Free user: Widget prikazuje "Free" sa gray badge, click otvara dropdown sa "Upgrade Plan" button
2. Pro user: Widget prikazuje "Pro" sa purple badge, dropdown pokazuje quota limits
3. Enterprise user: Widget prikazuje "Enterprise" sa gold badge, "Manage Subscription" button
4. Click na "Upgrade Plan" navigira na /subscription

**Success Criteria**:
- ✅ Prikazuje trenutni plan u NavBar-u
- ✅ Dropdown pokazuje quota limits za trenutni plan
- ✅ Različite boje za različite planove
- ✅ "Upgrade" button za Free/Basic, "Manage" za Pro/Enterprise

---

## 🧪 FAZA 3: TESTIRANJE (5h)

### T1: E2E Backend Testing ⚠️
**Prioritet**: HIGH
**Vrijeme**: 2h

**Test Suite 1: Training Quota Flow**

```bash
#!/bin/bash
# test_training_quota.sh

# Setup: Create test users with different plans
FREE_USER_TOKEN="..."
PRO_USER_TOKEN="..."
ENTERPRISE_USER_TOKEN="..."

echo "=== Test 1: Free User Cannot Train ==="
response=$(curl -s -X POST http://localhost:5000/api/training/train-models/test123 \
  -H "Authorization: Bearer $FREE_USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_parameters": {}}')

echo $response | jq .
# Expected: {"error": "Training not available", "plan": "Free"}

echo "\n=== Test 2: Pro User Within Limit ==="
response=$(curl -s -X POST http://localhost:5000/api/training/train-models/test456 \
  -H "Authorization: Bearer $PRO_USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_parameters": {}}')

echo $response | jq .
# Expected: {"success": true}

echo "\n=== Test 3: Check Usage Incremented ==="
response=$(curl -s -X GET http://localhost:5000/api/subscription/usage \
  -H "Authorization: Bearer $PRO_USER_TOKEN")

echo $response | jq .training_runs_count
# Expected: Previous count + 1

echo "\n=== Test 4: Pro User At Limit ==="
# Simulate 10 trainings already done
response=$(curl -s -X POST http://localhost:5000/api/training/train-models/test789 \
  -H "Authorization: Bearer $PRO_USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_parameters": {}}')

echo $response | jq .
# Expected: {"error": "Training limit reached", "current_usage": 10, "limit": 10}

echo "\n=== Test 5: Enterprise User Unlimited ==="
# Simulate 50 trainings already done
response=$(curl -s -X POST http://localhost:5000/api/training/train-models/test999 \
  -H "Authorization: Bearer $ENTERPRISE_USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_parameters": {}}')

echo $response | jq .
# Expected: {"success": true} - No limit
```

**Success Criteria**:
- ✅ Svi 5 testova prolaze sa expected response-ima
- ✅ Usage se inkrementira samo za uspješne training-e
- ✅ Free user NE MOŽE pokrenuti training
- ✅ Pro user može do limita, onda block
- ✅ Enterprise user nema limit

---

### T2: Frontend Integration Testing ⚠️
**Prioritet**: HIGH
**Vrijeme**: 2h

**Manual Test Checklist**:

**Scenario 1: Free User Journey**
1. ✅ Register/Login kao Free user
2. ✅ Navigate to Training page
3. ✅ QuotaWarningBanner prikazuje "Training not available"
4. ✅ "Train Models" button je disabled
5. ✅ Click na "Upgrade Plan" navigira na /subscription
6. ✅ Upgrade na Pro plan
7. ✅ Refresh stranice
8. ✅ QuotaCounter prikazuje "0/10 trainings"
9. ✅ "Train Models" button je enabled

**Scenario 2: Pro User At Limit**
1. ✅ Login kao Pro user sa 10/10 trainings
2. ✅ Navigate to Training page
3. ✅ QuotaCounter prikazuje "10/10 trainings" u red boji
4. ✅ QuotaWarningBanner prikazuje "Training limit reached"
5. ✅ "Train Models" button je disabled
6. ✅ Click na "Upgrade Plan" prikazuje Enterprise opcije

**Scenario 3: Pro User Normal Flow**
1. ✅ Login kao Pro user sa 5/10 trainings
2. ✅ Navigate to Training page
3. ✅ QuotaCounter prikazuje "5/10 trainings" u green boji
4. ✅ Configure model parameters
5. ✅ Click "Train Models"
6. ✅ Training pokreće se
7. ✅ Čekaj završetak training-a
8. ✅ QuotaCounter se automatski updatea na "6/10 trainings"
9. ✅ Repeat 4 more times
10. ✅ Na 10/10 - button postaje disabled

**Scenario 4: DataProcessingPage**
1. ✅ Login kao Free user sa 1/1 processing
2. ✅ Navigate to DataProcessingPage
3. ✅ QuotaWarningBanner prikazuje "Processing limit reached"
4. ✅ "Start Processing" button je disabled
5. ✅ Login kao Pro user sa 15/20 processing
6. ✅ "Start Processing" button je enabled
7. ✅ Click "Start Processing"
8. ✅ Processing radi
9. ✅ QuotaCounter updatea na 16/20

**Success Criteria**:
- ✅ Svi scenariji prolaze bez grešaka
- ✅ Usage se automatski osvježava nakon svake operacije
- ✅ Buttons su disabled/enabled ovisno o quota
- ✅ Warning banneri se prikazuju u pravim situacijama

---

### T3: Regression Testing ⚠️
**Prioritet**: MEDIUM
**Vrijeme**: 1h

**Provjeriti da postojeće funkcionalnosti još rade**:

1. ✅ Login/Logout flow
2. ✅ File upload bez quota checks
3. ✅ Generate datasets sa quota checks
4. ✅ Training sa quota checks
5. ✅ Model download
6. ✅ Session management
7. ✅ Language switching (DE/EN)

---

## 📋 Prioritizirani Task List

### ODMAH (Kritično - 3h)
1. ✅ **B1**: Dodati dekoratore na train_models endpoint (15 min)
2. ✅ **F1**: Usage refresh nakon training-a (1.5h)
3. ✅ **F2**: DataProcessingPage quota integracija (45 min)
4. ✅ **F3**: DataAdjustmentsPage quota integracija (45 min)

### USKORO (Važno - 6h)
5. ⚠️ **F4**: QuotaWarningBanner komponenta (2h)
6. ⚠️ **F5**: QuotaCounter komponenta (2h)
7. ⚠️ **F6**: SubscriptionStatus widget (2h)

### ZATIM (Testiranje - 5h)
8. ⚠️ **T1**: E2E Backend testing (2h)
9. ⚠️ **T2**: Frontend integration testing (2h)
10. ⚠️ **T3**: Regression testing (1h)

---

## 🎯 Ukupan Pregled

| Kategorija | Tasks | Vrijeme | Status |
|------------|-------|---------|--------|
| Backend Critical | 1 | 0.5h | ❌ Not Started |
| Frontend Critical | 3 | 3h | ❌ Not Started |
| Frontend Important | 3 | 6h | ❌ Not Started |
| Testing | 3 | 5h | ❌ Not Started |
| **TOTAL** | **10** | **14.5h** | **0% Complete** |

**Realistically**: Sa testiranjem i debugging-om, računaj **~16-18 sati** ukupno.

---

## 🚀 Sljedeći Koraci

1. **ODMAH**: Implementirati Task B1 (15 min) - dodati dekoratore
2. **ODMAH**: Testirati B1 sa curl komandama
3. **ZATIM**: Implementirati F1, F2, F3 u nizu (3h)
4. **ZATIM**: Testirati kritične flow-ove
5. **ONDA**: Implementirati F4, F5, F6 (6h)
6. **FINALNO**: Comprehensive testing (5h)

---

## 📝 Notes

- Backend je gotovo 95% kompletan - odličan posao!
- Frontend nedostaje ~15% - većinom UI komponente
- Najveći prioritet: **Zaštititi sve endpoints sa quota checks**
- Drugo najveći prioritet: **Usage refresh nakon operacija**
- QuotaWarningBanner i QuotaCounter su dependency za F2, F3 - implementiraj ih prije nego počneš F2/F3 ili koristi placeholder

---

## 🔗 Related Documents

- [Backend Comparison](./BACKEND_COMPARISON.md) - Detaljna analiza backend-a
- [Frontend Plan](./TRAINING_MONETIZATION_FRONTEND_PLAN.md) - Originalni frontend plan
- [Backend Plan](./TRAINING_MONETIZATION_BACKEND_PLAN.md) - Originalni backend plan
- [Phase 4 Progress](./phase4-progress.md) - Overall progress tracking
