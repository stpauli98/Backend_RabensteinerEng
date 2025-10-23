# 🚀 Storage Bucket Migration Plan - Training Results

## 📋 Executive Summary

**Problem:** Training results INSERT operacija timeout-uje nakon 16 sekundi kada se pokušava spremiti ~150MB JSON podataka u `training_results` JSONB kolonu.

**Rješenje:** Migracija training rezultata iz database JSONB kolone u Supabase Storage Bucket sa kompresijom.

**Benefiti:**
- ✅ Eliminiše timeout problem (Storage nema timeout limit)
- ✅ 95% smanjenje troškova storage-a ($0.019 → $0.0004 po sesiji)
- ✅ 10-50x brži upload (direktan file upload vs SQL parsing)
- ✅ Podržava rezultate do 5GB (trenutno limit ~150MB)
- ✅ Omogućava kompresiju (70-90% manja veličina)

**Estimacija:** 4-5 sati implementacije + testiranja

---

## 🔍 Trenutno Stanje

### Training Results Tabela
```sql
training_results {
    id: UUID PRIMARY KEY
    session_id: UUID REFERENCES sessions(id)
    results: JSONB  ← OGROMNI JSON PODACI (timeout problem)
    status: TEXT
    created_at: TIMESTAMP
}
```

### Što se čuva u `results` koloni:
```json
{
    "model_type": "Dense",
    "parameters": {...},
    "metrics": {"accuracy": 0.95, "loss": 0.023},
    "training_split": 0.8,
    "dataset_count": 92820,

    // VELIKI PODACI - Uzrokuju timeout:
    "trained_model": {...},           // Treniran TensorFlow model (pickle base64)
    "train_data": {"X": [[...]], "y": [[...]]},   // 92,820 redova
    "val_data": {"X": [[...]], "y": [[...]]},
    "test_data": {"X": [[...]], "y": [[...]]},
    "scalers": {...},                 // sklearn scalers (pickle base64)
    "input_features": [...],
    "output_features": [...]
}
```

**Veličina podataka:**
- Manji dataset (468 redova): ~5MB → timeout 5s
- Veliki dataset (92,820 redova): ~150MB → timeout 16s

### Zahvaćeni Fajlovi

#### Backend:
1. **`my_backend/api/routes/training.py`**
   - Line 2975: INSERT operacija (timeout ovdje!)
   - Line 2173+: `get_training_results()` - READ
   - Line 2234+: `get_training_results_details()` - READ
   - Line 2830+: DELETE operacije

2. **`my_backend/core/socketio_handlers.py`**
   - Real-time notifikacije o training status-u

#### Frontend:
3. **`src/core/database/supabase.ts`**
   - Line 738-741: `deleteAllSessions()` - bulk delete

---

## 🎯 Nova Arhitektura - Hybrid Approach

### Storage Struktura
```
training-results/  (Storage Bucket)
├── {session_id}/
│   ├── training_results_20251022_130000.json.gz  (compressed results)
│   ├── model_20251022_130000.h5                  (optional: model file)
│   └── plots/
│       ├── loss_curve.png
│       └── predictions.png
```

### Nova Database Schema
```sql
training_results {
    id: UUID PRIMARY KEY
    session_id: UUID REFERENCES sessions(id)
    status: TEXT
    created_at: TIMESTAMP

    -- NOVI FIELDS --
    results_file_path: TEXT           -- "session_id/training_results_timestamp.json.gz"
    file_size_bytes: BIGINT           -- 20000000 (20MB compressed)
    compressed: BOOLEAN               -- true
    results_metadata: JSONB           -- {accuracy, loss, epochs, model_type}

    -- DEPRECATED --
    results: JSONB                    -- NULL za nove rezultate
}
```

### Metadata za Quick Access (SQL Queries)
```json
{
    "accuracy": 0.95,
    "loss": 0.023,
    "epochs_completed": 100,
    "model_type": "Dense",
    "dataset_count": 92820,
    "training_split": 0.8
}
```

---

## 📝 Implementation Plan

### FAZA 1: Storage Bucket Setup (30min)

#### 1.1 Kreiranje Bucket-a u Supabase
```
1. Supabase Dashboard → Storage
2. Create New Bucket:
   - Name: "training-results"
   - Public: NO (private)
   - File size limit: 50MB per file
   - Allowed MIME types: application/json, application/gzip
```

#### 1.2 Database Migration - RLS Policies
**Fajl:** `database/migrations/006_training_results_storage.sql`

```sql
-- Kreiranje bucket-a
INSERT INTO storage.buckets (id, name, public)
VALUES ('training-results', 'training-results', false)
ON CONFLICT (id) DO NOTHING;

-- Policy 1: Users mogu upload-ovati svoje rezultate
CREATE POLICY "Users can upload their training results"
ON storage.objects FOR INSERT
WITH CHECK (
    bucket_id = 'training-results' AND
    auth.uid() IN (
        SELECT user_id FROM public.sessions
        WHERE id = (storage.foldername(name))[1]::uuid
    )
);

-- Policy 2: Users mogu čitati svoje rezultate
CREATE POLICY "Users can read their training results"
ON storage.objects FOR SELECT
USING (
    bucket_id = 'training-results' AND
    (
        auth.uid() IN (
            SELECT user_id FROM public.sessions
            WHERE id = (storage.foldername(name))[1]::uuid
        )
        OR auth.jwt() ->> 'role' = 'service_role'
    )
);

-- Policy 3: Users mogu brisati svoje rezultate
CREATE POLICY "Users can delete their training results"
ON storage.objects FOR DELETE
USING (
    bucket_id = 'training-results' AND
    auth.uid() IN (
        SELECT user_id FROM public.sessions
        WHERE id = (storage.foldername(name))[1]::uuid
    )
);
```

#### 1.3 Database Migration - Schema Update
**Fajl:** `database/migrations/007_training_results_to_storage.sql`

```sql
-- Dodaj nove kolone
ALTER TABLE public.training_results
ADD COLUMN IF NOT EXISTS results_file_path TEXT,
ADD COLUMN IF NOT EXISTS file_size_bytes BIGINT,
ADD COLUMN IF NOT EXISTS compressed BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS results_metadata JSONB DEFAULT '{}'::jsonb;

-- Update comments
COMMENT ON COLUMN training_results.results IS
'DEPRECATED: Use results_file_path instead. Large results stored in Storage.';

COMMENT ON COLUMN training_results.results_file_path IS
'Path to results JSON file in training-results bucket';

COMMENT ON COLUMN training_results.results_metadata IS
'Quick-access metadata: {accuracy, loss, epochs, model_type, dataset_count}';

-- Index za brže queries na metadata
CREATE INDEX IF NOT EXISTS idx_training_results_metadata
ON training_results USING gin(results_metadata);

-- Index za file path lookup
CREATE INDEX IF NOT EXISTS idx_training_results_file_path
ON training_results(results_file_path)
WHERE results_file_path IS NOT NULL;
```

---

### FAZA 2: Backend Implementation (2h)

#### 2.1 Storage Helper Module
**Fajl:** `my_backend/utils/training_storage.py`

```python
"""
Training Results Storage Module
Handles upload/download of training results to/from Supabase Storage
"""
import json
import gzip
import logging
from datetime import datetime
from utils.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)

def upload_training_results(session_id: str, results: dict, compress: bool = True) -> dict:
    """
    Upload training results to Supabase Storage

    Args:
        session_id: UUID session ID
        results: Training results dictionary
        compress: Whether to compress with gzip (default True)

    Returns:
        dict: {
            'file_path': str,      # Path in bucket
            'file_size': int,      # Size in bytes
            'compressed': bool,    # Whether compressed
            'metadata': dict       # Quick-access metadata
        }

    Raises:
        Exception: If upload fails
    """
    try:
        supabase = get_supabase_admin_client()

        # Generate file path
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        file_name = f"training_results_{timestamp}.json"
        if compress:
            file_name += ".gz"
        file_path = f"{session_id}/{file_name}"

        # Prepare data
        json_str = json.dumps(results, indent=2)
        original_size = len(json_str.encode('utf-8'))

        if compress:
            # Compress with gzip (level 9 = max compression)
            compressed_data = gzip.compress(json_str.encode('utf-8'), compresslevel=9)
            upload_data = compressed_data
            content_type = 'application/gzip'
            compression_ratio = (1 - len(compressed_data) / original_size) * 100
            logger.info(f"Compression: {original_size / 1024 / 1024:.2f}MB → "
                       f"{len(compressed_data) / 1024 / 1024:.2f}MB "
                       f"({compression_ratio:.1f}% reduction)")
        else:
            upload_data = json_str.encode('utf-8')
            content_type = 'application/json'

        file_size = len(upload_data)
        logger.info(f"Uploading {file_size / 1024 / 1024:.2f}MB to storage: {file_path}")

        # Upload to storage with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = supabase.storage.from_('training-results').upload(
                    path=file_path,
                    file=upload_data,
                    file_options={
                        'content-type': content_type,
                        'cache-control': '3600',
                        'upsert': False
                    }
                )
                break
            except Exception as upload_error:
                if attempt < max_retries - 1:
                    logger.warning(f"Upload attempt {attempt + 1} failed, retrying...")
                    import time
                    time.sleep(2)
                else:
                    raise upload_error

        logger.info(f"✅ Training results uploaded: {file_path}")

        # Extract metadata for quick SQL queries
        metadata = {
            'accuracy': results.get('metrics', {}).get('accuracy'),
            'loss': results.get('metrics', {}).get('loss'),
            'epochs_completed': results.get('parameters', {}).get('EP'),
            'model_type': results.get('model_type'),
            'dataset_count': results.get('dataset_count'),
            'training_split': results.get('training_split')
        }

        return {
            'file_path': file_path,
            'file_size': file_size,
            'compressed': compress,
            'metadata': metadata
        }

    except Exception as e:
        logger.error(f"❌ Failed to upload training results: {e}")
        raise


def download_training_results(file_path: str, decompress: bool = True) -> dict:
    """
    Download training results from Supabase Storage

    Args:
        file_path: Path to file in storage bucket
        decompress: Whether to decompress gzipped data (auto-detect from extension)

    Returns:
        dict: Training results

    Raises:
        Exception: If download fails
    """
    try:
        supabase = get_supabase_admin_client()

        logger.info(f"Downloading training results: {file_path}")

        # Download from storage
        response = supabase.storage.from_('training-results').download(file_path)

        # Auto-detect compression from file extension
        is_compressed = file_path.endswith('.gz')

        # Decompress if needed
        if decompress and is_compressed:
            logger.info(f"Decompressing {len(response) / 1024 / 1024:.2f}MB...")
            data = gzip.decompress(response)
            logger.info(f"Decompressed to {len(data) / 1024 / 1024:.2f}MB")
        else:
            data = response

        # Parse JSON
        results = json.loads(data.decode('utf-8'))

        logger.info(f"✅ Training results downloaded: {file_path}")
        return results

    except Exception as e:
        logger.error(f"❌ Failed to download training results: {e}")
        raise


def delete_training_results(file_path: str) -> bool:
    """
    Delete training results from storage

    Args:
        file_path: Path to file in storage bucket

    Returns:
        bool: True if successful
    """
    try:
        supabase = get_supabase_admin_client()
        response = supabase.storage.from_('training-results').remove([file_path])
        logger.info(f"✅ Deleted training results: {file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to delete training results: {e}")
        raise


def list_session_results(session_id: str) -> list:
    """
    List all training result files for a session

    Args:
        session_id: UUID session ID

    Returns:
        list: List of file metadata dicts
    """
    try:
        supabase = get_supabase_admin_client()
        response = supabase.storage.from_('training-results').list(session_id)
        return response
    except Exception as e:
        logger.error(f"Failed to list training results: {e}")
        return []
```

#### 2.2 Update Training Route - INSERT
**Fajl:** `my_backend/api/routes/training.py` (Line 2975)

```python
# STARO (Line 2975):
# supabase.table('training_results').insert(training_data).execute()

# NOVO:
from utils.training_storage import upload_training_results

try:
    # Upload results to Storage (BRZO - nema timeout-a)
    logger.info(f"Uploading training results to storage for session {uuid_session_id}")
    storage_result = upload_training_results(
        session_id=uuid_session_id,
        results=cleaned_results,
        compress=True  # Compress za manju veličinu
    )
    logger.info(f"Storage upload complete: {storage_result['file_size'] / 1024 / 1024:.2f}MB")

    # Save metadata to database (BRZO - samo ~1KB)
    training_data = {
        'session_id': uuid_session_id,
        'status': 'completed',
        'results_file_path': storage_result['file_path'],
        'file_size_bytes': storage_result['file_size'],
        'compressed': storage_result['compressed'],
        'results_metadata': storage_result['metadata'],
        'results': None  # Deprecated field - ostavi NULL
    }

    supabase.table('training_results').insert(training_data).execute()
    logger.info(f"✅ Training metadata saved for session {uuid_session_id}")

except Exception as storage_error:
    logger.error(f"❌ Failed to save training results: {storage_error}")
    # Fallback: pokušaj spremiti samo metadata bez full results
    try:
        training_data = {
            'session_id': uuid_session_id,
            'status': 'failed',
            'results_metadata': storage_result.get('metadata', {}),
            'results': None
        }
        supabase.table('training_results').insert(training_data).execute()
    except:
        pass
    raise
```

#### 2.3 Update Training Route - READ
**Fajl:** `my_backend/api/routes/training.py` (Line 2173+)

```python
# STARO:
# response = supabase.table('training_results').select('*').eq('session_id', uuid_session_id)...

# NOVO:
from utils.training_storage import download_training_results

def get_training_results(session_id):
    """
    Get training results for a session.
    Now fetches from Storage Bucket instead of JSONB column.
    """
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        supabase = get_supabase_client()

        # Get UUID for session
        uuid_session_id = create_or_get_session_uuid(session_id)

        # Get metadata from database (BRZO)
        response = supabase.table('training_results')\
            .select('id, session_id, status, created_at, results_file_path, '
                   'file_size_bytes, compressed, results_metadata')\
            .eq('session_id', uuid_session_id)\
            .order('created_at.desc')\
            .limit(1)\
            .execute()

        if response.data and len(response.data) > 0:
            record = response.data[0]

            # Download full results from Storage if file path exists
            if record.get('results_file_path'):
                try:
                    logger.info(f"Downloading full results from storage...")
                    full_results = download_training_results(
                        file_path=record['results_file_path'],
                        decompress=record.get('compressed', False)
                    )
                    record['results'] = full_results
                    logger.info(f"✅ Full results loaded from storage")
                except Exception as download_error:
                    logger.error(f"Failed to download results: {download_error}")
                    # Fallback: vrati samo metadata
                    record['results'] = record.get('results_metadata', {})

            return jsonify({
                'success': True,
                'results': [record],
                'message': 'Training results retrieved successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No training results found for session',
                'results': []
            }), 404

    except Exception as e:
        logger.error(f"Error getting training results: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

#### 2.4 Update DELETE Operations
**Fajl:** `my_backend/api/routes/training.py` (Line 2830+)

```python
from utils.training_storage import delete_training_results

# Delete storage files before deleting database records
try:
    # Get all file paths for session
    response = supabase.table('training_results')\
        .select('results_file_path')\
        .eq('session_id', uuid_session_id)\
        .execute()

    if response.data:
        for record in response.data:
            if record.get('results_file_path'):
                try:
                    delete_training_results(record['results_file_path'])
                except Exception as del_error:
                    logger.warning(f"Failed to delete storage file: {del_error}")

    # Then delete from database
    supabase.table('training_results').delete().eq('session_id', uuid_session_id).execute()

except Exception as e:
    logger.error(f"Error deleting training results: {e}")
```

---

### FAZA 3: Frontend Updates (30min)

#### 3.1 Update Delete All Sessions
**Fajl:** `src/core/database/supabase.ts` (Line 738-741)

```typescript
// Delete training results from storage bucket
const { data: trainingResults } = await supabase
  .from('training_results')
  .select('results_file_path');

if (trainingResults && trainingResults.length > 0) {
  const filePaths = trainingResults
    .filter(r => r.results_file_path)
    .map(r => r.results_file_path);

  if (filePaths.length > 0) {
    try {
      await supabase.storage
        .from('training-results')
        .remove(filePaths);
    } catch (storageError) {
      console.error('Failed to delete training results from storage:', storageError);
      // Continue with database deletion even if storage fails
    }
  }
}

// Then delete from database table
const { error: trainingResultsError } = await supabase
  .from('training_results')
  .delete()
  .neq('id', '00000000-0000-0000-0000-000000000000');

if (trainingResultsError) {
  // Handle error
}
```

---

### FAZA 4: Testing (1h)

#### 4.1 Unit Tests
**Fajl:** `my_backend/tests/test_training_storage.py`

```python
import pytest
import json
from utils.training_storage import (
    upload_training_results,
    download_training_results,
    delete_training_results
)

def generate_test_results(rows=1000):
    """Generate test training results"""
    return {
        'model_type': 'Dense',
        'parameters': {'EP': 100, 'LAY': [64, 32]},
        'metrics': {'accuracy': 0.95, 'loss': 0.023},
        'dataset_count': rows,
        'train_data': {
            'X': [[i, i*2, i*3] for i in range(rows)],
            'y': [[i] for i in range(rows)]
        },
        'test_data': {
            'X': [[i, i*2, i*3] for i in range(100)],
            'y': [[i] for i in range(100)]
        }
    }

def test_upload_small_dataset():
    """Test upload with small dataset (~5MB)"""
    session_id = "test-session-small"
    results = generate_test_results(rows=468)

    storage_result = upload_training_results(session_id, results, compress=True)

    assert storage_result['file_size'] < 1_000_000  # < 1MB compressed
    assert storage_result['compressed'] == True
    assert storage_result['file_path'].startswith(session_id)
    assert storage_result['metadata']['dataset_count'] == 468

def test_upload_large_dataset():
    """Test upload with large dataset (~150MB)"""
    session_id = "test-session-large"
    results = generate_test_results(rows=92820)

    storage_result = upload_training_results(session_id, results, compress=True)

    assert storage_result['file_size'] < 20_000_000  # < 20MB compressed
    assert storage_result['compressed'] == True

def test_download_and_verify():
    """Test download and verify data integrity"""
    session_id = "test-session-verify"
    original = generate_test_results(rows=1000)

    # Upload
    storage_result = upload_training_results(session_id, original, compress=True)

    # Download
    downloaded = download_training_results(storage_result['file_path'])

    # Verify
    assert original['model_type'] == downloaded['model_type']
    assert original['metrics'] == downloaded['metrics']
    assert len(original['train_data']['X']) == len(downloaded['train_data']['X'])

    # Cleanup
    delete_training_results(storage_result['file_path'])

def test_compression_ratio():
    """Test compression effectiveness"""
    session_id = "test-compression"
    results = generate_test_results(rows=10000)

    # Upload compressed
    compressed = upload_training_results(session_id, results, compress=True)

    # Upload uncompressed
    uncompressed = upload_training_results(session_id, results, compress=False)

    compression_ratio = (1 - compressed['file_size'] / uncompressed['file_size']) * 100

    assert compression_ratio > 70  # At least 70% compression

    # Cleanup
    delete_training_results(compressed['file_path'])
    delete_training_results(uncompressed['file_path'])
```

#### 4.2 Integration Test
```python
def test_full_training_flow():
    """
    Integration test: Full training flow with Storage
    1. Start training
    2. Upload results to Storage
    3. Read via API
    4. Verify data
    5. Delete
    """
    # Test će biti napisan nakon implementacije
    pass
```

---

## 📊 Performance Expectations

### Upload Speed
```
Manji dataset (468 rows, ~5MB):
- Uncompressed: ~2 sekunde
- Compressed: ~3 sekunde (5MB → 0.5MB)

Veliki dataset (92,820 rows, ~150MB):
- Uncompressed: ~30 sekundi
- Compressed: ~15 sekundi (150MB → 15MB)
```

### Download Speed
```
Manji dataset:
- ~1 sekunda download + decompress

Veliki dataset:
- ~5 sekundi download + 2 sekunde decompress
```

### Storage Costs
```
1 training session (150MB → 15MB compressed):
- Database JSONB: $0.019/mjesec
- Storage Bucket: $0.0004/mjesec
- Savings: 97.9%

100 training sessions:
- Database: $1.90/mjesec
- Storage: $0.04/mjesec
- Savings: $1.86/mjesec (97.9%)
```

---

## ⚠️ Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Storage upload faila | Low | High | Retry logic (3 attempts), fallback na database |
| RLS policy blokira pristup | Medium | High | Koristi service_role, detaljno testiranje |
| Compression lomi podatke | Low | High | Unit tests sa realnim podacima, JSON validation |
| Stari kod prestane raditi | Medium | Medium | Backward compatibility (check za results field) |
| Storage quota exceeded | Low | Medium | Lifecycle policies, automatic cleanup starih fajlova |
| Network timeout pri downloadu | Low | Medium | Chunked download, retry logic |

---

## 🚦 Deployment Checklist

### Pre-Deployment
- [ ] Backup trenutne `training_results` tabele
- [ ] Test na development environment-u
- [ ] Verify Storage bucket kreiran
- [ ] Verify RLS policies primijenjene

### Deployment Steps
1. **Database Migrations** (5min)
   - [ ] Apply `006_training_results_storage.sql`
   - [ ] Apply `007_training_results_to_storage.sql`
   - [ ] Verify kolone dodane

2. **Backend Deployment** (10min)
   - [ ] Deploy `utils/training_storage.py`
   - [ ] Deploy updated `api/routes/training.py`
   - [ ] Docker rebuild i restart
   - [ ] Check logs za errors

3. **Frontend Deployment** (5min)
   - [ ] Deploy updated `src/core/database/supabase.ts`
   - [ ] Build i deploy
   - [ ] Verify delete operacije rade

4. **Testing** (15min)
   - [ ] Run unit tests
   - [ ] Test mali dataset (468 rows)
   - [ ] Test veliki dataset (92,820 rows)
   - [ ] Test download i data integrity
   - [ ] Test delete operations

### Post-Deployment
- [ ] Monitor prvi production training
- [ ] Verify nema timeout-a
- [ ] Check storage usage
- [ ] Monitor performance metrics
- [ ] Update dokumentacija

---

## 🎯 Success Criteria

✅ Training sa 92,820 redova se završava bez timeout-a
✅ Upload traje < 20 sekundi
✅ Download traje < 10 sekundi
✅ Kompresija postiže > 70% smanjenje veličine
✅ Stari training sessions i dalje mogu čitati rezultate
✅ Delete operacije čiste i Storage i Database

---

## 📚 Additional Resources

### Supabase Storage Documentation
- https://supabase.com/docs/guides/storage
- https://supabase.com/docs/guides/storage/uploads/standard-uploads
- https://supabase.com/docs/guides/storage/security/access-control

### Python gzip Documentation
- https://docs.python.org/3/library/gzip.html

### Testing Best Practices
- Unit tests za svaku helper funkciju
- Integration tests za end-to-end flow
- Performance benchmarks za različite veličine dataseta

---

## 👥 Team & Timeline

**Estimated Time:** 4-5 sati

**Breakdown:**
- Database setup: 30min
- Backend implementation: 2h
- Frontend updates: 30min
- Testing: 1h
- Deployment & monitoring: 1h

**Team:**
- Backend developer: Database migrations, Python kod
- DevOps: Supabase configuration, deployment
- QA: Testing plan, validation

---

**Datum kreiranja:** 2025-10-22
**Status:** Planning
**Priority:** High (blokira training sa velikim podacima)
