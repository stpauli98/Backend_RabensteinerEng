# TensorFlow Upgrade 2.16.1 → 2.19.0 — Design Spec

**Date:** 2026-04-14
**Status:** Approved for planning
**Owner:** Backend
**Scope:** `Bekend/my_backend/` only

## Context

The backend currently uses `tensorflow==2.16.1` with a `numpy<2.0` pin (TF 2.16 is incompatible with numpy 2.x). Models are saved in legacy `.h5` HDF5 format. A user attempting to load a backend-trained model locally with TF 2.20.0 received:

```
Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
```

Root cause: Keras 3.0 (in TF 2.16) embeds `quantization_config` into Dense layer config when serializing. Newer Keras 3.x versions (in TF 2.19+/2.20) renamed/removed that argument, breaking forward compatibility of existing `.h5` artifacts.

Existing trained models in Supabase Storage are disposable — they will be overwritten by new training runs.

## Goals

- Upgrade `tensorflow` from 2.16.1 to 2.19.0
- Switch model save format from `.h5` to `.keras` (native Keras 3 zip format)
- Eliminate `quantization_config` deserialization errors
- Remove the `numpy<2.0` upper-bound pin
- Verify all 7 model training types still work (Dense, CNN, LSTM, AR-LSTM, SVR, Linear, LGBMR)

## Non-Goals

- Migrating existing `.h5` models in Supabase Storage (disposable; will be overwritten)
- Backward compatibility with old `.h5` files
- Updating local consumer scripts (e.g. `func_load.py`) — owner installs TF 2.19.0 locally on their own
- Refactoring training code or any other module
- Bumping unrelated dependency versions

## Approach

**Big-bang switch in a single feature branch.** Justified because:
- Existing models are disposable (no migration work)
- Code uses only stable `tf.keras.*` API surface (Sequential, Dense, LSTM, Conv2D, standard callbacks/optimizers/losses) — no breaking changes between Keras 3.0 and 3.8
- `model_storage.py` already supports both `.h5` and `.keras` extensions in its loader
- Existing `tests/test_upgrade_smoke.py` provides validation infrastructure
- Docker image gives clean rollback (revert PR + redeploy)

## Code Changes

### `requirements.txt`

```diff
- numpy>=1.26.3,<2.0  # MUST stay <2.0 - TensorFlow 2.16 crashes with numpy 2.0+
+ numpy>=1.26.3,<3.0
- tensorflow==2.16.1
+ tensorflow==2.19.0
```

### `domains/training/ml/trainer.py`

Four save-path strings change `.h5` → `.keras`:
- Line 498: `dense_{dataset_name}_{timestamp}.h5` → `.keras`
- Line 513: `cnn_{dataset_name}_{timestamp}.h5` → `.keras`
- Line 528: `lstm_{dataset_name}_{timestamp}.h5` → `.keras`
- Line 543: `ar_lstm_{dataset_name}_{timestamp}.h5` → `.keras`

### `domains/training/ml/integration.py`

Three save-path strings change `.h5` → `.keras`:
- Line 411: dense path
- Line 432: cnn path
- Line 453: lstm path

### `utils/model_storage.py`

- Update docstrings on lines 3, 31, 35, 41, 54, 59 to refer to `.keras` instead of `.h5`
- Verify upload `content_type` parameter works for `.keras` files; set to `application/octet-stream` if Supabase Storage rejects the default
- Loader on line 203 already supports both extensions — no change

### Untouched

- All `tf.keras.*` API call sites in `trainer.py`, `integration.py`, `exact.py`, `socketio.py`, `prediction_service.py`, `models.py` — API is stable across Keras 3.0 → 3.8
- `Dockerfile` — Python 3.11 base supports TF 2.19.0
- `.pkl` save paths for SVR, Linear, LGBMR (joblib-based, not TF)

**Total: 11 line edits, 1 dependency bump.**

## Testing Strategy

### Pre-Upgrade Baseline (TF 2.16.1)

1. Run `pytest tests/test_upgrade_smoke.py` — record result
2. Run full `pytest` suite — record pass/fail counts
3. Note baseline numbers in PR description

### Post-Upgrade Verification (TF 2.19.0)

**Level 1 — Unit/smoke (Docker):**
- `pytest tests/test_upgrade_smoke.py` must pass
- Full `pytest` suite — pass count must not regress below baseline

**Level 2 — Training pipeline (Docker, end-to-end via API):**

Train one small model of each type using a test dataset (short epochs, small subset):

| Type | Format | Validation |
|---|---|---|
| Dense | `.keras` | Train OK → file written → upload OK → download + load OK → predict OK |
| CNN | `.keras` | Same as above |
| LSTM | `.keras` | Same as above |
| AR-LSTM | `.keras` | Same as above |
| SVR | `.pkl` | Control group — must not regress |
| Linear | `.pkl` | Control group — must not regress |
| LGBMR | `.pkl` | Control group — must not regress |

For each Keras model:
- Training completes without errors or new warnings
- File saved as `.keras` in `uploads/trained_models/`
- Upload to Supabase Storage succeeds
- Download from Supabase + `load_model()` succeeds
- `model.predict()` returns valid output shape

**Level 3 — WebSocket progress:**
Run a Dense training and verify `SocketIOProgressCallback` emits `training_status_update` events to the session room.

**Level 4 — Plot generation:**
`POST /api/training/generate-plot` must produce a valid plot using a freshly-trained model (validates `i_dat_inf`/`o_dat_inf` metadata flow).

### Acceptance Criteria

- All 4 verification levels pass
- No Keras serialization, deprecation, or quantization warnings in logs
- `numpy.__version__` in container resolves to 2.x
- Model files on disk and in Supabase Storage have `.keras` extension

## Rollout

1. Create feature branch `feature/tf-upgrade-2.19`
2. Apply all edits in a single atomic commit
3. Build Docker image with `.env`: `docker build --build-arg ENV_FILE=.env -t my_backend .`
4. Run container locally, execute Levels 2–4 manually
5. Push branch, request code review
6. Merge to `main` after review approval
7. Deploy to staging if available, otherwise direct production
8. Monitor logs for 24h: scan for `Keras`, `deserial`, `quantization`, `warning`

## Rollback

**Trigger:** Any Level 2/3/4 failure in staging, or production crash on training endpoints.

**Procedure:**
1. `git revert <merge-commit>` → push → redeploy previous Docker image
2. Note: any `.keras` models written to Supabase Storage between deploy and rollback become unloadable on TF 2.16.1; users must retrain (acceptable since models are disposable)

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Subtle Keras 3 API change not caught by code review | Low | Medium | Smoke-test all 7 model types before merge |
| numpy 2.x breaks transitive dependency (lightgbm/sklearn/matplotlib) | Low | Low | Run pytest suite; if it fails, re-pin `numpy<2.0` (TF 2.19 supports both) |
| Supabase Storage MIME-type issue for `.keras` files | Low | Low | Set `content_type='application/octet-stream'` in upload call |
| `SocketIOProgressCallback` breaks due to Keras 3.x callback signature change | Very Low | Medium | Manual training test with WebSocket monitoring (Level 3) |
| Plot generation breaks due to model metadata change | Low | Low | Verify Level 4 test |

## Open Questions

None at design time. Surface during implementation if Keras 3.8 introduces a runtime-only behavior delta vs Keras 3.0.

## References

- `Bekend/my_backend/CLAUDE.md` — backend architecture
- `Bekend/my_backend/requirements.txt` — current pins
- `Bekend/my_backend/domains/training/ml/trainer.py` — 4 save sites
- `Bekend/my_backend/domains/training/ml/integration.py` — 3 save sites
- `Bekend/my_backend/utils/model_storage.py` — upload/download with dual-extension support
- `Bekend/my_backend/tests/test_upgrade_smoke.py` — existing smoke test
