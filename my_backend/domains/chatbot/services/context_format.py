"""Sanitize and format the optional chat `context` (current settings + data profile)."""

_MAX_FIELDS = 20
_MAX_COLS = 40
_MAX_STR = 120
_SCAN_LIMIT = 200  # upper bound on how many raw items we scan before capping


def _s(v) -> str:
    return str(v)[:_MAX_STR]


def sanitize_context(raw):
    """Return a capped, validated context dict, or None if there's nothing usable."""
    if not isinstance(raw, dict):
        return None
    out = {}

    fields = raw.get("fields")
    if isinstance(fields, list):
        clean = []
        for f in fields[:_SCAN_LIMIT]:  # bound the scan, keep up to _MAX_FIELDS valid items
            if isinstance(f, dict) and isinstance(f.get("label"), str) and isinstance(f.get("value"), str):
                clean.append({"label": f["label"][:_MAX_STR], "value": f["value"][:_MAX_STR]})
                if len(clean) >= _MAX_FIELDS:
                    break
        if clean:
            out["fields"] = clean

    dp = raw.get("dataProfile")
    if isinstance(dp, dict):
        prof = {}
        if isinstance(dp.get("rowCount"), int):
            prof["rowCount"] = max(0, min(dp["rowCount"], 10 ** 9))
        cols = dp.get("columns")
        if isinstance(cols, list):
            prof["columns"] = [_s(c) for c in cols[:_MAX_COLS] if isinstance(c, str)]
        tc = dp.get("timeColumn")
        if isinstance(tc, dict):
            res = tc.get("resolutionMinutes")
            prof["timeColumn"] = {
                "resolutionMinutes": res if isinstance(res, int) else None,
                "rangeStart": _s(tc.get("rangeStart", "")),
                "rangeEnd": _s(tc.get("rangeEnd", "")),
            }
        cq = dp.get("columnQuality")
        if isinstance(cq, list):
            q = []
            for item in cq[:_MAX_COLS]:
                if isinstance(item, dict) and isinstance(item.get("column"), str):
                    q.append({
                        "column": item["column"][:_MAX_STR],
                        "missingPct": int(item["missingPct"]) if isinstance(item.get("missingPct"), (int, float)) else 0,
                        "zerosPct": int(item["zerosPct"]) if isinstance(item.get("zerosPct"), (int, float)) else 0,
                    })
            if q:
                prof["columnQuality"] = q
        if prof:
            out["dataProfile"] = prof

    return out or None


def format_context(context) -> str:
    """Render the context as plain text for the volatile system tail."""
    if not context:
        return ""
    parts = []
    fields = context.get("fields")
    if fields:
        lines = "\n".join(f"- {f['label']}: {f['value']}" for f in fields)
        parts.append("The user's current settings on this step:\n" + lines)
    dp = context.get("dataProfile")
    if dp:
        seg = [f"The user's loaded data: {dp.get('rowCount', '?')} rows"]
        if dp.get("columns"):
            seg.append("columns [" + ", ".join(dp["columns"]) + "]")
        tc = dp.get("timeColumn")
        if tc:
            res = tc.get("resolutionMinutes")
            seg.append(f"detected time resolution {res} min" if res is not None else "time resolution not detected")
            seg.append(f"range {tc.get('rangeStart', '')}-{tc.get('rangeEnd', '')}")
        parts.append("; ".join(seg) + ".")
        gaps = [q for q in (dp.get("columnQuality") or []) if q.get("missingPct") or q.get("zerosPct")]
        if gaps:
            parts.append("Columns with gaps: " + "; ".join(
                f"{q['column']} {q['missingPct']}% missing / {q['zerosPct']}% zeros" for q in gaps))
    return "\n\n".join(parts)
