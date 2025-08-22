# Training Module - Database Integration Fixes

## Critical Issues Found

### 1. ZEITSCHRITTE Table - offset/offsett inconsistency
**Problem**: Database column is named `offsett` (double 't') but frontend uses `offset`

**Fix Options**:
- Option A: Rename database column from `offsett` to `offset` 
- Option B: Update frontend to use `offsett` consistently

**Recommended**: Option A - Fix database to use standard naming

```sql
-- Database migration to fix offsett column
ALTER TABLE zeitschritte 
RENAME COLUMN offsett TO offset;
```

### 2. FILES Table - Naming Convention Mismatch
**Problem**: Frontend uses camelCase, database uses snake_case

**Current Mapping**:
```
Frontend (camelCase)          → Database (snake_case)
numerischeDatenpunkte        → numerische_datenpunkte
numerischerAnteil            → numerischer_anteil  
zeitschrittweiteAvgValue     → zeitschrittweite_mittelwert
zeitschrittweiteMinValue     → zeitschrittweite_min
mittelwertbildungÜberDenZeithorizont → mittelwertbildung_uber_den_zeithorizont
```

**Backend handles this with conversion in save_file_info():**
```python
# Line 383-389 in database.py
"zeitschrittweite_mittelwert": str(file_info.get("zeitschrittweiteAvgValue", file_info.get("zeitschrittweiteMittelwert", ""))),
"zeitschrittweite_min": str(file_info.get("zeitschrittweiteMinValue", file_info.get("zeitschrittweiteMin", ""))),
```

### 3. FILES Table - zeithorizont field issue
**Problem**: Frontend sends separate fields but backend expects single field

**Current State**:
- Frontend sends: `zeithorizontStart`, `zeithorizontEnd`
- Database has: `zeithorizont_start`, `zeithorizont_end` ✅
- Backend Python correctly maps these fields

### 4. Working Correctly ✅

- **Session ID mapping**: String to UUID conversion works properly
- **TIME_INFO JSONB structure**: Properly implemented with category_data
- **Training tables**: training_progress, training_results, training_logs all properly structured

## Recommendations

1. **Standardize naming conventions**:
   - Use snake_case consistently in database
   - Use camelCase consistently in frontend
   - Have clear conversion layer in backend

2. **Fix the offsett column** in zeitschritte table (Priority: HIGH)

3. **Document all field mappings** clearly in code

4. **Add TypeScript interfaces** that match database schema exactly:
```typescript
// Database-aligned interfaces
interface ZeitschritteDB {
  session_id: string;
  eingabe: string;
  ausgabe: string;
  zeitschrittweite: string;
  offset: string; // After fixing database column
}
```

## Current Status
- ✅ Session mapping works
- ✅ Time info JSONB structure works
- ⚠️ File metadata field naming needs attention
- ❌ zeitschritte.offsett needs to be renamed to offset
- ✅ Training results and progress tables properly structured