# Idempotent threshold endpoints — auto-recompute intermediate state

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the user-facing 409 "STL intermediate state missing — re-run /start" (and the LSTM twin) by having the threshold endpoints transparently recompute the missing intermediate result instead of refusing the request.

**Architecture:** Today, `POST /api/adjustmentsOfData/stl-threshold` requires `state["intermediate"]["stl_result"]` to be present; if missing it returns 409. The STL decomposition is deterministic from `state["processed_df"]` + `state["params"]["STL"]["var"]["PERIOD"]`, both of which are already in the surviving state when this 409 fires. The fix is to add two helpers (`_ensure_stl_intermediate`, `_ensure_lstm_intermediate`) that lazily recompute the intermediate when absent and write it back to state, then call them from both threshold endpoints before the existing None check. The visible error becomes a transparent cache miss.

**Tech Stack:** Flask, Python 3.9, pytest with the existing Flask app fixture (`tests/domains/adjustments/test_anomaly_endpoints.py`), Docker for runtime verification (`docker compose up --build -d` from `Bekend/`).

---

## Branch + repo state

Backend repo at `/Users/nmil/Desktop/Posao2/Bekend`, currently on `fix/expose-internal-errors-dev` with unrelated untracked files. New branch must be created off `main` (or `origin/main`) so this work merges cleanly.

## File structure

| File | Role | Touched |
|---|---|---|
| `my_backend/domains/adjustments/api/adjustments.py` | Flask endpoint module — STL/LSTM threshold handlers + new private helpers | modified |
| `my_backend/tests/domains/adjustments/test_anomaly_endpoints.py` | pytest module exercising endpoints via Flask app | modified (2 new tests) |
| `my_backend/docs/superpowers/plans/2026-05-13-idempotent-threshold-endpoints.md` | this plan | created |

No new modules, no new files beyond the plan doc. The helpers live alongside the existing endpoint handlers because they share private state-shape assumptions (`state["processed_df"]`, `state["intermediate"]`, `state["params"]`).

## Key code references

- `adjustments.py:651` — imports `prepare_stl as _prepare_stl`, `prepare_lstm as _prepare_lstm`
- `adjustments.py:1023-1031` — original `/start` STL compute path (template for the recovery helper)
- `adjustments.py:1051-1064` — original `/start` LSTM compute path (template for the recovery helper)
- `adjustments.py:1146-1150` — STL threshold endpoint: status check + intermediate None check (call site for `_ensure_stl_intermediate`)
- `adjustments.py:1270-1277` — LSTM threshold endpoint: same shape (call site for `_ensure_lstm_intermediate`)
- `services/state_manager.py:122-143` — anomaly state schema documentation

## Existing tests that must keep passing

- `test_stl_threshold_valid_completes` — happy path
- `test_stl_threshold_invalid_returns_400_de` — input validation
- `test_stl_threshold_wrong_state_returns_409` — pipeline-status mismatch (this 409 path remains; only the *missing-intermediate* 409 disappears)
- `test_stl_threshold_lstm_nan_keeps_pipeline_retryable` — error-recovery semantic

---

## Task 1: Branch off main

**Files:** none — pure git operation.

- [ ] **Step 1: From repo root, fetch latest main and branch off it.**

```bash
cd /Users/nmil/Desktop/Posao2/Bekend
git fetch origin main
git checkout -b fix/idempotent-threshold-endpoints origin/main
```

- [ ] **Step 2: Verify the branch is clean and based on main.**

```bash
git status
git log --oneline -3
```

Expected: `working tree clean`, top commit is whatever is current on `origin/main`.

---

## Task 2: Failing test for STL intermediate auto-recompute

**Files:**
- Modify: `my_backend/tests/domains/adjustments/test_anomaly_endpoints.py` (add new test after `test_stl_threshold_valid_completes`, around line 423)

The test runs the full flow up to AWAITING_STL_THRESHOLD, then manually wipes `state["intermediate"]["stl_result"]` to simulate the TTL/restart loss, then submits a valid threshold and asserts the request still succeeds with 200.

- [ ] **Step 1: Write the failing test.**

Append to `tests/domains/adjustments/test_anomaly_endpoints.py`:

```python
def test_stl_threshold_recovers_when_intermediate_lost(app, staged_test2):
    """If state.intermediate.stl_result has been wiped (TTL / process restart),
    the threshold endpoint must recompute it transparently rather than
    returning a 409 that the user has no clear path to recover from."""
    from domains.adjustments.services.state_manager import _get_anomaly_state

    upload_id, _ = staged_test2
    _load_and_start(app, upload_id, _default_params(stl_run=True, lstm_run=False))

    # Simulate the production failure mode: pipeline is still in
    # AWAITING_STL_THRESHOLD but intermediate was cleared (e.g. selective
    # cleanup, debugger session, future cache eviction).
    with app.test_request_context():
        # The state manager exposes the same in-process dict via _get_anomaly_state
        # used by the endpoint handler.
        state = _get_anomaly_state(upload_id)
        assert state is not None, "fixture must leave the state alive"
        assert state["intermediate"]["stl_result"] is not None
        state["intermediate"]["stl_result"] = None

    r = _post_json(app, adj_module.anomaly_stl_threshold,
                   "/api/adjustmentsOfData/stl-threshold",
                   {"uploadId": upload_id, "threshold": 50, "lang": "en"})
    assert _status(r) == 200, _body(r)
    body = _body(r)
    assert body["status"] == PipelineStatus.COMPLETE
    assert "stlAnomalies" in body["plots"]
```

- [ ] **Step 2: Confirm the test imports + helpers work — discover the right state-access helper if `_get_anomaly_state` doesn't exist verbatim.**

Run: `grep -n "def _get_anomaly\|def get_anomaly" my_backend/domains/adjustments/services/state_manager.py`

Expected: a function name like `_get_anomaly_state(upload_id)` returning the state dict (used at `adjustments.py:1143` as `_get_anomaly`). If the public name differs, fix the import in the test to match.

- [ ] **Step 3: Run the test, expect FAIL with 409.**

```bash
cd my_backend
docker compose -f ../docker-compose.yml run --rm backend \
  pytest tests/domains/adjustments/test_anomaly_endpoints.py::test_stl_threshold_recovers_when_intermediate_lost -v
```

Expected output contains:
```
AssertionError: ... assert 409 == 200
```

(If running tests outside Docker is the established convention, fall back to `pytest tests/...` from `my_backend/`. Use whichever convention the existing test files document.)

---

## Task 3: Implement `_ensure_stl_intermediate` and call it from the STL threshold endpoint

**Files:**
- Modify: `my_backend/domains/adjustments/api/adjustments.py`
  - Add helper near other private helpers — search for `def _get_anomaly` or `def _try_acquire_pipeline` to find the cluster; place above them.
  - Modify endpoint body at line 1148-1150.

- [ ] **Step 1: Add the helper.**

Insert this helper near the other private helpers in `adjustments.py` (above the `anomaly_stl_threshold` route handler so it's visible at call time — exact line will depend on existing helper placement, search for `def _get_anomaly` to find the cluster):

```python
def _ensure_stl_intermediate(state, lang):
    """Recompute state['intermediate']['stl_result'] if it has been wiped.

    The threshold endpoint used to return 409 "STL intermediate state
    missing — re-run /start" in this case, forcing the user to start over.
    STL decomposition is deterministic from `state['processed_df']` plus
    `state['params']['STL']['var']['PERIOD']`, both of which survive the
    same TTL/restart events that can lose the intermediate. Recovery is
    therefore local and free of side effects on other endpoints.
    """
    if state.get("intermediate", {}).get("stl_result") is not None:
        return
    processed_df = state.get("processed_df")
    par = state.get("params") or {}
    period_descriptor = (par.get("STL", {}).get("var", {}).get("PERIOD") or {})
    period_raw = period_descriptor.get("value")
    if processed_df is None or period_raw is None:
        # Nothing to recompute from — let the caller emit the original 409.
        return
    period = int(period_raw)
    stl_result, _time_values = _prepare_stl(processed_df, period, lang=lang)
    state.setdefault("intermediate", {})["stl_result"] = stl_result
```

- [ ] **Step 2: Wire the helper into the endpoint.**

In `anomaly_stl_threshold` (around line 1148), change:

```python
        if state.get("pipeline_status") != _PipelineStatus.AWAITING_STL_THRESHOLD:
            return jsonify({"error": "Pipeline is not awaiting an STL threshold"}), 409
        stl_result = state.get("intermediate", {}).get("stl_result")
        if stl_result is None:
            return jsonify({"error": "STL intermediate state missing — re-run /start"}), 409
```

to:

```python
        if state.get("pipeline_status") != _PipelineStatus.AWAITING_STL_THRESHOLD:
            return jsonify({"error": "Pipeline is not awaiting an STL threshold"}), 409
        try:
            _ensure_stl_intermediate(state, lang)
        except ValueError as e:
            state["pipeline_status"] = _PipelineStatus.ERROR
            return jsonify({"error": str(e)}), 400
        stl_result = state.get("intermediate", {}).get("stl_result")
        if stl_result is None:
            # Recovery wasn't possible (processed_df or params evicted) —
            # original 409 path stands.
            return jsonify({"error": "STL intermediate state missing — re-run /start"}), 409
```

- [ ] **Step 3: Run the new test alone, expect PASS.**

```bash
cd my_backend
docker compose -f ../docker-compose.yml run --rm backend \
  pytest tests/domains/adjustments/test_anomaly_endpoints.py::test_stl_threshold_recovers_when_intermediate_lost -v
```

Expected output contains: `PASSED`.

- [ ] **Step 4: Run the existing STL threshold test suite, expect all green.**

```bash
docker compose -f ../docker-compose.yml run --rm backend \
  pytest tests/domains/adjustments/test_anomaly_endpoints.py -v -k "stl_threshold"
```

Expected: `test_stl_threshold_invalid_returns_400_de`, `test_stl_threshold_valid_completes`, `test_stl_threshold_wrong_state_returns_409`, `test_stl_threshold_lstm_nan_keeps_pipeline_retryable`, `test_stl_threshold_recovers_when_intermediate_lost` all PASSED.

- [ ] **Step 5: Commit.**

```bash
git add my_backend/domains/adjustments/api/adjustments.py \
        my_backend/tests/domains/adjustments/test_anomaly_endpoints.py
git commit -m "fix(adjustments): recompute STL intermediate when missing in /stl-threshold

The endpoint used to return 409 \"STL intermediate state missing — re-run /start\"
when state['intermediate']['stl_result'] had been cleared (TTL eviction,
process restart, or any future cache pressure). STL decomposition is
deterministic from state['processed_df'] + state['params'], both of which
survive the same loss events. New _ensure_stl_intermediate helper
recomputes on demand and writes back to state. Existing 409 path remains
as a last-resort fallback if processed_df itself is gone.

Tests: +1 (test_stl_threshold_recovers_when_intermediate_lost); existing
STL threshold tests still pass."
```

---

## Task 4: Failing test for LSTM intermediate auto-recompute

**Files:**
- Modify: `my_backend/tests/domains/adjustments/test_anomaly_endpoints.py` (append a sibling test mirroring Task 2's shape but for LSTM).

- [ ] **Step 1: Write the failing test.**

Append to the test file (look for any existing LSTM threshold test to anchor placement; if there are none, place after the STL recovery test):

```python
def test_lstm_threshold_recovers_when_intermediate_lost(app, staged_test2):
    """Symmetric to the STL recovery: if intermediate.lstm_results_df is
    missing, the /lstm-threshold endpoint must recompute rather than 409."""
    from domains.adjustments.services.state_manager import _get_anomaly_state

    upload_id, _ = staged_test2
    # stl_run=False so the pipeline pauses at AWAITING_LSTM_THRESHOLD after /start.
    _load_and_start(app, upload_id, _default_params(stl_run=False, lstm_run=True))

    with app.test_request_context():
        state = _get_anomaly_state(upload_id)
        assert state is not None
        assert state["intermediate"]["lstm_results_df"] is not None
        state["intermediate"]["lstm_results_df"] = None

    r = _post_json(app, adj_module.anomaly_lstm_threshold,
                   "/api/adjustmentsOfData/lstm-threshold",
                   {"uploadId": upload_id, "threshold": 50, "lang": "en"})
    assert _status(r) == 200, _body(r)
    body = _body(r)
    assert body["status"] == PipelineStatus.COMPLETE
    assert "lstmAnomalies" in body["plots"]
```

- [ ] **Step 2: Verify the route handler name matches.**

Run: `grep -n "anomaly_lstm_threshold" my_backend/domains/adjustments/api/adjustments.py`

Expected: a function with that exact name. If the symbol is different (e.g. `anomaly_lstm_submit`), correct the `adj_module.anomaly_lstm_threshold` reference in the test.

- [ ] **Step 3: Run the test, expect FAIL with 409.**

```bash
cd my_backend
docker compose -f ../docker-compose.yml run --rm backend \
  pytest tests/domains/adjustments/test_anomaly_endpoints.py::test_lstm_threshold_recovers_when_intermediate_lost -v
```

Expected: `AssertionError: ... assert 409 == 200`.

---

## Task 5: Implement `_ensure_lstm_intermediate` and call it from the LSTM threshold endpoint

**Files:**
- Modify: `my_backend/domains/adjustments/api/adjustments.py`
  - Add helper next to `_ensure_stl_intermediate`.
  - Modify endpoint body at line 1275-1277.

- [ ] **Step 1: Add the helper.**

Insert below `_ensure_stl_intermediate`:

```python
def _ensure_lstm_intermediate(state, lang):
    """Recompute state['intermediate']['lstm_results_df'] if it has been
    wiped — same idempotency contract as _ensure_stl_intermediate."""
    if state.get("intermediate", {}).get("lstm_results_df") is not None:
        return
    processed_df = state.get("processed_df")
    par = state.get("params") or {}
    lstm = par.get("LSTM", {}).get("var", {})
    period_raw = (lstm.get("PERIOD") or {}).get("value")
    neurons = (lstm.get("NEURONS") or {}).get("value")
    epochs = (lstm.get("EPOCHS") or {}).get("value")
    batch_size = (lstm.get("BATCH_SIZE") or {}).get("value")
    if processed_df is None or None in (period_raw, neurons, epochs, batch_size):
        return
    period = int(period_raw)
    results_df, _model = _prepare_lstm(
        processed_df, period, neurons, epochs, batch_size, lang=lang,
    )
    state.setdefault("intermediate", {})["lstm_results_df"] = results_df
```

- [ ] **Step 2: Wire the helper into the endpoint.**

In `anomaly_lstm_threshold` (around line 1275), change:

```python
        if state.get("pipeline_status") != _PipelineStatus.AWAITING_LSTM_THRESHOLD:
            return jsonify({"error": "Pipeline is not awaiting an LSTM threshold"}), 409
        results_df = state.get("intermediate", {}).get("lstm_results_df")
        if results_df is None:
            return jsonify({"error": "LSTM intermediate state missing — re-run /start"}), 409
```

to:

```python
        if state.get("pipeline_status") != _PipelineStatus.AWAITING_LSTM_THRESHOLD:
            return jsonify({"error": "Pipeline is not awaiting an LSTM threshold"}), 409
        try:
            _ensure_lstm_intermediate(state, lang)
        except ValueError as e:
            state["pipeline_status"] = _PipelineStatus.ERROR
            return jsonify({"error": str(e)}), 400
        results_df = state.get("intermediate", {}).get("lstm_results_df")
        if results_df is None:
            return jsonify({"error": "LSTM intermediate state missing — re-run /start"}), 409
```

- [ ] **Step 3: Run the new test alone, expect PASS.**

```bash
docker compose -f ../docker-compose.yml run --rm backend \
  pytest tests/domains/adjustments/test_anomaly_endpoints.py::test_lstm_threshold_recovers_when_intermediate_lost -v
```

Expected: `PASSED`.

- [ ] **Step 4: Run the full anomaly endpoint test file.**

```bash
docker compose -f ../docker-compose.yml run --rm backend \
  pytest tests/domains/adjustments/test_anomaly_endpoints.py -v
```

Expected: all tests PASSED (no regressions in STL valid path, LSTM valid path, wrong-state 409s, or NaN-retryable behaviour).

- [ ] **Step 5: Commit.**

```bash
git add my_backend/domains/adjustments/api/adjustments.py \
        my_backend/tests/domains/adjustments/test_anomaly_endpoints.py
git commit -m "fix(adjustments): recompute LSTM intermediate when missing in /lstm-threshold

Symmetric to the STL recovery: when state['intermediate']['lstm_results_df']
has been wiped, recompute it from processed_df + LSTM params instead of
returning the dead-end 409. The original 409 path remains for the (rare)
case where processed_df itself has been evicted.

Tests: +1 (test_lstm_threshold_recovers_when_intermediate_lost)."
```

---

## Task 6: Wider regression run + Docker smoke

- [ ] **Step 1: Run the full adjustments test suite.**

```bash
docker compose -f ../docker-compose.yml run --rm backend \
  pytest tests/domains/adjustments -v
```

Expected: all tests PASSED. The new tests should appear in the output. Pay attention to: `test_stl_threshold_*`, `test_lstm_threshold_*`, `test_anomaly_pipeline*` if any exists.

- [ ] **Step 2: Build + start the backend in Docker and confirm it boots cleanly.**

```bash
cd /Users/nmil/Desktop/Posao2/Bekend
docker compose up --build -d
docker compose ps
docker logs bekend-backend-1 --tail 30
```

Expected: `Up (healthy)`, no traceback in logs, gunicorn listening on 8080.

- [ ] **Step 3: Sanity-check from outside the container.**

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/health
```

Expected: `200`.

---

## Task 7: Plan doc commit + push branch

- [ ] **Step 1: Add the plan doc and commit.**

```bash
cd /Users/nmil/Desktop/Posao2/Bekend
git add my_backend/docs/superpowers/plans/2026-05-13-idempotent-threshold-endpoints.md
git commit -m "docs(adjustments): plan — idempotent STL/LSTM threshold endpoints"
```

- [ ] **Step 2: Push the branch.**

```bash
git push -u origin fix/idempotent-threshold-endpoints
```

Expected output: GitHub URL for opening a PR.

---

## Self-review

**Spec coverage check.** The spec asks for: (1) eliminate user-facing 409 for missing STL intermediate, (2) same for LSTM, (3) reuse the `/start` compute path so behaviour matches what `/start` would have produced, (4) Docker-based testing per project convention, (5) new branch off main. All five covered — STL in Tasks 2-3, LSTM in Tasks 4-5, reuse via the shared `_prepare_stl` / `_prepare_lstm` imports (`adjustments.py:661`), Docker test commands in every test step + Task 6, branch creation in Task 1.

**Placeholder scan.** No `TBD`, `TODO`, `implement later`, or vague "add error handling" steps. Each code block is complete and self-contained. The two `if processed_df is None or … is None: return` guard branches are intentional and documented — they preserve the original 409 as a last-resort fallback when even `processed_df` is gone (which would mean the entire state has effectively decayed).

**Type / symbol consistency.** Helper names `_ensure_stl_intermediate` / `_ensure_lstm_intermediate` are used consistently in implementation and call-site steps. Test names `test_stl_threshold_recovers_when_intermediate_lost` / `test_lstm_threshold_recovers_when_intermediate_lost` mirror existing test conventions. The state-access helper `_get_anomaly_state` may need a name-correction step (Task 2 Step 2, Task 4 Step 2) — both tasks include a grep verification.

**Risk audit.**
- Mutation of state in the recompute helpers is intentional and matches how `/start` writes the same keys (`adjustments.py:1031`, `:1064`).
- New tests bypass the public start endpoint when wiping the intermediate, so we can't double-set the status — the existing pipeline-status check at line 1146/1273 still fires correctly for the wrong-state 409 path.
- No threading concerns: `_try_acquire_pipeline` is called downstream of the new recompute branch (`adjustments.py:1166`), so the pipeline lock is acquired only after the intermediate is guaranteed present, identical to the legacy flow.
