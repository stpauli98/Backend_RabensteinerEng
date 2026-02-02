# ToDo - Backend/Frontend Issues

## High Priority

### 1. Frontend progress stuck at 20% after training completes
- **Status:** Partially Fixed ✅
- **Date:** 2026-01-30
- **Fix Date:** 2026-01-30
- **Description:** After model training completes, the frontend progress bar stays at 20% and doesn't update. User sees "Dense training completed in 2m 23s" but progress doesn't advance.
- **Root cause:** Frontend only handled `training_completed` and `post_training` statuses. For `model_training_completed`, it fell back to epoch-based calculation (20/100 epochs = 20% due to early stopping).
- **Frontend Fix (commit c9e5d5a):** Updated `RealTimeTrainingProgress.tsx` to:
  - Handle `model_training_completed` status
  - Prioritize `progress_percent` from backend over epoch calculation
  - Show post-training phases (50-100%) correctly
- **Remaining work:** Backend needs to emit more SocketIO events during post-training phases. Currently emits at 50%, 55%, 60-95% but these might not reach frontend due to Cloud Run WebSocket limitations.

### 2. Evaluation progress not visible on frontend
- **Status:** Open (related to #1)
- **Date:** 2026-01-30
- **Description:** During training, the evaluation phase (averaging, metrics calculation) has no progress indicator on frontend.
- **Backend logs show:**
  ```
  Starting evaluation with averaging: n_max=12, O_N=97
  Averaging complete. Calculating overall metrics...
  Overall metrics complete. Calculating per-timestep metrics...
  Per-timestep metrics complete. Generating df_eval...
  df_eval complete. Generating df_eval_ts...
  Evaluation complete.
  ```
- **Frontend shows:** Nothing - stuck at same progress

### 3. Post-training save/upload progress not visible
- **Status:** Open (related to #1)
- **Date:** 2026-01-30
- **Description:** After evaluation completes, the results are saved to Supabase storage. This can take time but user sees no progress.
- **Phases not shown:**
  - Preparing results for pickle serialization
  - Compressing data (gzip)
  - Uploading to Supabase storage
  - Training pipeline completed

---

## Medium Priority

### 4. Add "Processing..." indicator for long operations
- **Status:** Open
- **Description:** When backend is processing but not emitting progress, frontend should show a generic "Processing..." indicator instead of appearing frozen.

---

## Completed

- [x] Migrate JSON to Pickle format for training results
- [x] Add optimized transformer with feature flags (156x faster!)
- [x] Add dual-format deserialization for backward compatibility
- [x] Deploy to Cloud Run with ENV variables
- [x] Fix frontend progress calculation to use backend's progress_percent (commit c9e5d5a)

---

## Notes

### Current training phases (backend):
1. Dataset generation (transformer) - ~30s with optimization
2. Scaling datasets
3. Model training (epochs) - 2-5 min depending on early stopping
4. Model testing/predictions
5. Inverse scaling
6. Evaluation with 12-level averaging
7. Results preparation (pickle serialization)
8. Upload to Supabase storage
9. Pipeline complete

### SocketIO events that SHOULD be emitted:
- `training_progress` - for each phase
- `training_epoch` - for each epoch
- `training_complete` - when done
- `training_error` - on failure
