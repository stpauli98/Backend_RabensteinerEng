# TensorFlow Upgrade 2.16.1 → 2.19.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade backend `tensorflow` 2.16.1 → 2.19.0, switch model save format `.h5` → `.keras`, remove `numpy<2.0` pin, verify all 7 training types work.

**Architecture:** Single feature branch, atomic commit. 11 line edits + 1 dependency bump. Existing `.h5` models in Supabase Storage are disposable (overwritten by new training runs). Validation via existing `tests/test_upgrade_smoke.py` plus end-to-end manual training of all 7 model types.

**Tech Stack:** Python 3.11 (Docker base), Flask, TensorFlow 2.19, Keras 3, numpy 2.x, Supabase Storage, joblib (for `.pkl` models).

**Spec:** `docs/superpowers/specs/2026-04-14-tensorflow-upgrade-2.19-design.md`

---

## File Structure

| File | Action | What changes |
|---|---|---|
| `requirements.txt` | Modify | tensorflow pin + numpy pin |
| `domains/training/ml/trainer.py` | Modify | 4 save-path strings: `.h5` → `.keras` |
| `domains/training/ml/integration.py` | Modify | 3 save-path strings: `.h5` → `.keras` |
| `utils/model_storage.py` | Modify | Docstrings only (loader and uploader handle extension dynamically) |

No new files. No file splits.

---

### Task 1: Capture pre-upgrade baseline

**Files:**
- Read: `tests/test_upgrade_smoke.py` (existing)

- [ ] **Step 1: Build current Docker image**

Run from `Bekend/my_backend/`:
```bash
docker build --build-arg ENV_FILE=.env -t my_backend:pre_upgrade .
```
Expected: image builds successfully with `tensorflow==2.16.1`.

- [ ] **Step 2: Run smoke tests inside container, capture output**

```bash
docker run --rm --env-file .env my_backend:pre_upgrade pytest tests/test_upgrade_smoke.py -v 2>&1 | tee /tmp/baseline_smoke.txt
```
Expected: record pass/fail counts. Save the file path for reference in PR description.

- [ ] **Step 3: Run full pytest suite, capture pass/fail counts**

```bash
docker run --rm --env-file .env my_backend:pre_upgrade pytest 2>&1 | tail -20 | tee /tmp/baseline_full.txt
```
Expected: note the final `===== X passed, Y failed, Z skipped =====` line — this is the regression baseline.

- [ ] **Step 4: Verify current TF and numpy versions**

```bash
docker run --rm --env-file .env my_backend:pre_upgrade python -c "import tensorflow as tf, numpy as np; print(f'tf={tf.__version__} np={np.__version__}')"
```
Expected: `tf=2.16.1 np=1.26.x`

---

### Task 2: Create feature branch

**Files:** none (git only)

- [ ] **Step 1: Verify clean working tree**

Run from `Bekend/`:
```bash
git status
git branch --show-current
```
Expected: clean tree, currently on `main` (or whatever default branch is).

- [ ] **Step 2: Create and switch to feature branch**

```bash
git checkout -b feature/tf-upgrade-2.19
```
Expected: `Switched to a new branch 'feature/tf-upgrade-2.19'`.

---

### Task 3: Update `requirements.txt`

**Files:**
- Modify: `Bekend/my_backend/requirements.txt:16,24`

- [ ] **Step 1: Apply edits**

In `Bekend/my_backend/requirements.txt`, change line 16:

From:
```
numpy>=1.26.3,<2.0  # MUST stay <2.0 - TensorFlow 2.16 crashes with numpy 2.0+
```
To:
```
numpy>=1.26.3,<3.0
```

And line 24:

From:
```
tensorflow==2.16.1
```
To:
```
tensorflow==2.19.0
```

- [ ] **Step 2: Verify diff**

```bash
git diff Bekend/my_backend/requirements.txt
```
Expected: exactly 2 lines changed (numpy and tensorflow), no other modifications.

---

### Task 4: Update model save paths in `trainer.py`

**Files:**
- Modify: `Bekend/my_backend/domains/training/ml/trainer.py:498,513,528,543`

- [ ] **Step 1: Replace `.h5` with `.keras` in 4 save paths**

In `Bekend/my_backend/domains/training/ml/trainer.py`:

Line 498:
From: `model_path = os.path.join(models_dir, f'dense_{dataset_name}_{timestamp}.h5')`
To:   `model_path = os.path.join(models_dir, f'dense_{dataset_name}_{timestamp}.keras')`

Line 513:
From: `model_path = os.path.join(models_dir, f'cnn_{dataset_name}_{timestamp}.h5')`
To:   `model_path = os.path.join(models_dir, f'cnn_{dataset_name}_{timestamp}.keras')`

Line 528:
From: `model_path = os.path.join(models_dir, f'lstm_{dataset_name}_{timestamp}.h5')`
To:   `model_path = os.path.join(models_dir, f'lstm_{dataset_name}_{timestamp}.keras')`

Line 543:
From: `model_path = os.path.join(models_dir, f'ar_lstm_{dataset_name}_{timestamp}.h5')`
To:   `model_path = os.path.join(models_dir, f'ar_lstm_{dataset_name}_{timestamp}.keras')`

- [ ] **Step 2: Verify only 4 lines changed and no `.h5` remain in this file**

```bash
git diff Bekend/my_backend/domains/training/ml/trainer.py | grep -E "^[+-]" | grep -v "^[+-]{3}"
```
Expected: 8 lines (4 removed + 4 added), all `.h5` → `.keras` substitutions.

```bash
grep -n "\.h5" Bekend/my_backend/domains/training/ml/trainer.py
```
Expected: no output (no remaining `.h5` references).

---

### Task 5: Update model save paths in `integration.py`

**Files:**
- Modify: `Bekend/my_backend/domains/training/ml/integration.py:411,432,453`

- [ ] **Step 1: Replace `.h5` with `.keras` in 3 save paths**

In `Bekend/my_backend/domains/training/ml/integration.py`:

Line 411:
From: `model_path = os.path.join(models_dir, f'dense_{dataset_name}_{timestamp}.h5')`
To:   `model_path = os.path.join(models_dir, f'dense_{dataset_name}_{timestamp}.keras')`

Line 432:
From: `model_path = os.path.join(models_dir, f'cnn_{dataset_name}_{timestamp}.h5')`
To:   `model_path = os.path.join(models_dir, f'cnn_{dataset_name}_{timestamp}.keras')`

Line 453:
From: `model_path = os.path.join(models_dir, f'lstm_{dataset_name}_{timestamp}.h5')`
To:   `model_path = os.path.join(models_dir, f'lstm_{dataset_name}_{timestamp}.keras')`

- [ ] **Step 2: Verify**

```bash
git diff Bekend/my_backend/domains/training/ml/integration.py | grep -E "^[+-]" | grep -v "^[+-]{3}"
```
Expected: 6 lines (3 removed + 3 added).

```bash
grep -n "\.h5" Bekend/my_backend/domains/training/ml/integration.py
```
Expected: no output.

---

### Task 6: Update docstrings in `model_storage.py`

**Files:**
- Modify: `Bekend/my_backend/utils/model_storage.py:3,31,35,41,54,59`

The uploader and loader handle the file extension dynamically (`os.path.splitext()` on line 70 and `endswith()` check on line 203), so functional code requires no change. Only docstrings reference `.h5` literally.

- [ ] **Step 1: Update file-level docstring (line 3)**

From: `Handles upload/download of trained models (.h5 files) to/from Supabase Storage`
To:   `Handles upload/download of trained models (.keras files) to/from Supabase Storage`

- [ ] **Step 2: Update `upload_trained_model` docstring**

Line 31:
From: `Upload a trained model (.h5 file) to Supabase Storage`
To:   `Upload a trained model (.keras file) to Supabase Storage`

Line 35:
From: `model_file_path: Local path to the .h5 model file`
To:   `model_file_path: Local path to the .keras model file`

Line 41:
From: `'file_path': str,       # Path in bucket: "session_id/model_type_dataset_timestamp.h5"`
To:   `'file_path': str,       # Path in bucket: "session_id/model_type_dataset_timestamp.keras"`

Line 54:
From: `...     model_file_path="/app/uploads/trained_models/dense_dataset1_20251023.h5",`
To:   `...     model_file_path="/app/uploads/trained_models/dense_dataset1_20251023.keras",`

Line 59:
From: `"abc-123/dense_dataset1_20251023_123456.h5"`
To:   `"abc-123/dense_dataset1_20251023_123456.keras"`

- [ ] **Step 3: Verify only docstring lines changed**

```bash
git diff Bekend/my_backend/utils/model_storage.py
```
Expected: 6 docstring/comment lines changed (3, 31, 35, 41, 54, 59). No code logic changes.

Note: leave the literal extension checks on lines 203, 205, 263, 325, 328 unchanged — they handle both `.h5` and `.keras` and continue to support legacy reads if any old file slips through.

---

### Task 7: Build new Docker image

**Files:** none (Docker build context)

- [ ] **Step 1: Build with `.env`**

Run from `Bekend/my_backend/`:
```bash
docker build --build-arg ENV_FILE=.env -t my_backend:tf_2.19 .
```
Expected: image builds successfully. Look out for any pip resolution errors involving numpy / tensorflow / lightgbm / scikit-learn.

If pip fails: capture the error, do not proceed. Likely cause = transitive dep conflict; check `lightgbm` and `scikit-learn` versions against TF 2.19 / numpy 2.x.

- [ ] **Step 2: Verify versions inside new image**

```bash
docker run --rm --env-file .env my_backend:tf_2.19 python -c "import tensorflow as tf, numpy as np, keras; print(f'tf={tf.__version__} np={np.__version__} keras={keras.__version__}')"
```
Expected: `tf=2.19.0 np=2.x.x keras=3.x.x`

---

### Task 8: Run smoke tests against new image

**Files:**
- Read: `tests/test_upgrade_smoke.py`

- [ ] **Step 1: Run smoke test suite in new container**

```bash
docker run --rm --env-file .env my_backend:tf_2.19 pytest tests/test_upgrade_smoke.py -v 2>&1 | tee /tmp/post_smoke.txt
```
Expected: all tests that passed in baseline (`/tmp/baseline_smoke.txt`) must still pass.

- [ ] **Step 2: Diff against baseline**

```bash
diff /tmp/baseline_smoke.txt /tmp/post_smoke.txt
```
Expected: no test moves from PASS to FAIL. New PASSes are acceptable. Any regression → STOP and investigate before proceeding.

- [ ] **Step 3: Run full pytest suite**

```bash
docker run --rm --env-file .env my_backend:tf_2.19 pytest 2>&1 | tail -20 | tee /tmp/post_full.txt
```
Expected: pass count `≥` baseline count from `/tmp/baseline_full.txt`. Any regression → STOP and investigate.

---

### Task 9: End-to-end training verification — Dense

**Files:** none (manual API testing against running container)

This task validates Level 2 (training pipeline) for the Dense model type. Repeat the pattern in Tasks 10–12 for CNN, LSTM, AR-LSTM.

- [ ] **Step 1: Start container with port mapping**

```bash
docker run -d --name backend_tf219_test -p 8080:8080 --env-file .env my_backend:tf_2.19
sleep 5
curl -f http://localhost:8080/health
```
Expected: `{"status":"ok"}` from health endpoint.

- [ ] **Step 2: Train a Dense model via API**

Use the standard training workflow (upload session → process → train) with a small test CSV (≤1000 rows) and short epochs (≤5) to keep test fast. Trigger Dense training via `POST /api/training/...` (use the existing Postman/frontend flow to construct the request).

Expected:
- 200 response from training endpoint
- WebSocket `training_status_update` events received (Level 3 validation in same step)
- Container logs show `model.save(...)` called with a `.keras` path
- Logs contain no `Keras`, `deserial`, `quantization`, or `Deprecation` warnings

- [ ] **Step 3: Verify file written with `.keras` extension**

```bash
docker exec backend_tf219_test sh -c 'ls -la uploads/trained_models/ 2>/dev/null'
```
Expected: at least one `dense_*_*.keras` file present.

- [ ] **Step 4: Verify upload to Supabase Storage**

Check Supabase Studio → Storage → `trained-models` bucket → session_id folder. Must contain a `dense_*_*.keras` file matching the filename from Step 3.

- [ ] **Step 5: Verify load + predict round-trip**

Trigger any endpoint that calls `load_model_from_storage` for the just-trained model (e.g., prediction endpoint, or use the Python REPL inside the container):

```bash
docker exec -it backend_tf219_test python -c "
from utils.model_storage import load_model_from_storage
import numpy as np
# Replace SESSION_ID and FILENAME with values from previous steps
m = load_model_from_storage('SESSION_ID', 'FILENAME.keras')
print('Loaded:', type(m).__name__, 'input shape:', m.input_shape)
# Smoke predict with zeros of correct shape
shape = list(m.input_shape)
shape[0] = 1
out = m.predict(np.zeros(shape))
print('Predict OK, output shape:', out.shape)
"
```
Expected: prints `Loaded: Sequential ...` and `Predict OK, ...` without exceptions or `quantization_config` errors.

- [ ] **Step 6: Stop and remove test container before next task**

```bash
docker stop backend_tf219_test && docker rm backend_tf219_test
```

---

### Task 10: End-to-end training verification — CNN

**Files:** none

Repeat Task 9 verbatim, substituting `cnn` for `dense`:

- [ ] **Step 1: Start container** (same command as Task 9, Step 1, container name `backend_tf219_test`)
- [ ] **Step 2: Train CNN via API** (use CNN training endpoint instead of Dense; expect `.keras` save and clean logs)
- [ ] **Step 3: Verify file** (`ls uploads/trained_models/` shows `cnn_*_*.keras`)
- [ ] **Step 4: Verify Supabase upload** (check `trained-models` bucket for `cnn_*_*.keras`)
- [ ] **Step 5: Verify load + predict** (same Python snippet as Task 9 Step 5, with new SESSION_ID/FILENAME)
- [ ] **Step 6: Stop and remove container**

Expected outcomes identical to Task 9 — file extension `.keras`, no Keras warnings, load + predict round-trip succeeds.

---

### Task 11: End-to-end training verification — LSTM

**Files:** none

Repeat Task 9 verbatim, substituting `lstm` for `dense`:

- [ ] **Step 1: Start container**
- [ ] **Step 2: Train LSTM via API**
- [ ] **Step 3: Verify file** (`lstm_*_*.keras` in `uploads/trained_models/`)
- [ ] **Step 4: Verify Supabase upload** (`lstm_*_*.keras` in bucket)
- [ ] **Step 5: Verify load + predict**
- [ ] **Step 6: Stop and remove container**

---

### Task 12: End-to-end training verification — AR-LSTM

**Files:** none

Repeat Task 9 verbatim, substituting `ar_lstm` for `dense`:

- [ ] **Step 1: Start container**
- [ ] **Step 2: Train AR-LSTM via API**
- [ ] **Step 3: Verify file** (`ar_lstm_*_*.keras` in `uploads/trained_models/`)
- [ ] **Step 4: Verify Supabase upload** (`ar_lstm_*_*.keras` in bucket)
- [ ] **Step 5: Verify load + predict**
- [ ] **Step 6: Stop and remove container**

---

### Task 13: End-to-end verification — control group (SVR, Linear, LGBMR)

**Files:** none

These models use joblib `.pkl` format and are NOT touched by this upgrade. Verify they still work to rule out collateral damage from numpy 2.x.

- [ ] **Step 1: Start container**

```bash
docker run -d --name backend_tf219_test -p 8080:8080 --env-file .env my_backend:tf_2.19
sleep 5
```

- [ ] **Step 2: Train SVR via API**

Expected: training succeeds, `svr_*_*.pkl` file written, no errors.

- [ ] **Step 3: Train Linear via API**

Expected: `linear_*_*.pkl` written, no errors.

- [ ] **Step 4: Train LGBMR via API**

Expected: training succeeds, no `numpy` or `lightgbm` errors in logs.

- [ ] **Step 5: Stop and remove container**

```bash
docker stop backend_tf219_test && docker rm backend_tf219_test
```

---

### Task 14: Plot generation verification

**Files:** none (uses one of the `.keras` models from Tasks 9–12)

Validates Level 4 from spec: `i_dat_inf`/`o_dat_inf` metadata flow + plot rendering with new model format.

- [ ] **Step 1: Start container**

```bash
docker run -d --name backend_tf219_test -p 8080:8080 --env-file .env my_backend:tf_2.19
sleep 5
```

- [ ] **Step 2: Re-train one Dense model** (so we have fresh training results in the active session)

Use the same flow as Task 9 Step 2.

- [ ] **Step 3: Call plot endpoint**

```bash
curl -X POST http://localhost:8080/api/training/generate-plot \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <JWT>" \
  -d '{"session_id": "SESSION_ID", "model_type": "dense", "dataset_name": "DATASET"}' \
  -o /tmp/plot.png -w "%{http_code}\n"
```
Expected: HTTP 200, `/tmp/plot.png` is a valid PNG > 1KB.

```bash
file /tmp/plot.png
```
Expected: `PNG image data, ...`.

- [ ] **Step 4: Stop and remove container**

```bash
docker stop backend_tf219_test && docker rm backend_tf219_test
```

---

### Task 15: Commit and push

**Files:** all modified files from Tasks 3–6.

- [ ] **Step 1: Stage exact files**

```bash
cd Bekend
git add my_backend/requirements.txt \
        my_backend/domains/training/ml/trainer.py \
        my_backend/domains/training/ml/integration.py \
        my_backend/utils/model_storage.py \
        my_backend/docs/superpowers/specs/2026-04-14-tensorflow-upgrade-2.19-design.md \
        my_backend/docs/superpowers/plans/2026-04-14-tensorflow-upgrade-2.19.md
```

- [ ] **Step 2: Verify staged diff**

```bash
git diff --staged --stat
```
Expected: 4 source files + 2 doc files. No unrelated files.

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
chore(backend): upgrade tensorflow 2.16.1 → 2.19.0, switch to .keras format

- Bump tensorflow==2.19.0 in requirements.txt
- Remove numpy<2.0 upper-bound pin (TF 2.19 supports numpy 2.x)
- Switch all 4 Keras model save paths in trainer.py from .h5 to .keras
- Switch all 3 Keras model save paths in integration.py from .h5 to .keras
- Update model_storage.py docstrings (.h5 → .keras); functional code unchanged
- Loader still accepts both .h5 and .keras, so legacy reads remain safe

Validation:
- Smoke tests: pass ≥ baseline
- E2E training: Dense, CNN, LSTM, AR-LSTM all produce .keras, load + predict OK
- Control group: SVR, Linear, LGBMR (.pkl) unaffected
- Plot generation: works against new .keras models

Old .h5 models in Supabase Storage become unloadable — acceptable per spec
(disposable, overwritten on next training run).

Spec: docs/superpowers/specs/2026-04-14-tensorflow-upgrade-2.19-design.md
Plan: docs/superpowers/plans/2026-04-14-tensorflow-upgrade-2.19.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Push branch**

```bash
git push -u origin feature/tf-upgrade-2.19
```
Expected: branch published, ready for code review.

---

### Task 16: Post-deploy monitoring (after merge)

**Files:** none (production observability)

- [ ] **Step 1: Watch container logs for first 24 hours**

After merge to `main` and production deploy, scan logs:

```bash
docker logs <prod_container> 2>&1 | grep -iE "keras|deserial|quantization|deprecat" | head -50
```
Expected: no matches. Any match → investigate immediately, prepare rollback if it indicates training breakage.

- [ ] **Step 2: Verify first user-triggered training writes `.keras`**

After first real user training in production, check Supabase Storage `trained-models` bucket. New file must end in `.keras`.

- [ ] **Step 3: If failure detected — rollback**

```bash
git revert <merge-commit-sha>
git push origin main
```
Then redeploy. Spec section "Rollback" documents the recovery sequence.

---

## Self-Review Notes

**Spec coverage:**
- Goals (1) tf upgrade → Tasks 3, 7
- Goals (2) format switch → Tasks 4, 5
- Goals (3) eliminate `quantization_config` errors → Tasks 9 Step 5, 10–12 Step 5 (load + predict round-trip)
- Goals (4) numpy unpin → Task 3
- Goals (5) verify all 7 model types → Tasks 9 (Dense), 10 (CNN), 11 (LSTM), 12 (AR-LSTM), 13 (SVR + Linear + LGBMR)
- Testing Levels 1–4 → Tasks 8, 9–13, 9 Step 2 (WebSocket), 14 (Plot)
- Rollout → Task 7 (build), 15 (commit/push)
- Rollback → Task 16 Step 3
- Risks: numpy 2.x dep break → Task 7 Step 1 catch + Task 8 + Task 13; SocketIO callback → Task 9 Step 2; plot metadata → Task 14; MIME type → covered by `model_storage.py` dynamic extension handling

**Placeholders:** none — every step has concrete commands or exact diffs. SESSION_ID / FILENAME placeholders in Tasks 9–12, 14 Step 3 are runtime values from prior steps, which is expected.

**Type/path consistency:** all line numbers cross-checked against current files. All file extensions consistent (`.keras` for new Keras models, `.pkl` for sklearn/lightgbm — unchanged).
