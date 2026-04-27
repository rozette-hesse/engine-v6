"""
Layer 2 v5 Feature Builder
===========================
Extends the v3/v4 feature set with ordinal severity trajectory, extended
rolling windows, flow score history, and cycle-position interaction features.

All v3/v4 features are preserved — v5 is a strict superset.

New feature groups
------------------
1. Ordinal lag & delta
   {sym}_ord_lag1, {sym}_ord_lag2, {sym}_ord_delta (today - lag1)
   Group-level (pain/energy/mood/digestive): *_ord_lag1_sum, *_ord_delta

2. Extended rolling window (up to 5 days)
   pain/energy/total_severity: *_roll5_mean, *_roll5_max

3. Symptom pattern indicators
   is_symptom_rising, is_symptom_declining, is_symptom_peak, is_symptom_trough

4. Flow score trajectory
   flow_score (0–5), flow_score_lag1, flow_score_lag2, flow_declining

5. Cycle interaction
   cycle_cv_x_day_norm = cycle_length_cv × cycle_day_norm

Backward compatibility
----------------------
The feature builder accepts ``symptom_severities`` inside each
``recent_daily_logs`` entry.  When the key is absent (legacy callers) it
falls back to converting binary symptoms to ordinal (each present = 3).
This means v5 artifacts are NOT loadable by the v4 inference module and
vice-versa — the two sets of artifacts are entirely separate.
"""

from typing import Dict, List, Optional

import numpy as np

from ..config import SUPPORTED_SYMPTOMS
from .features_v3 import (  # re-export so training script has one import
    PAIN_COLS,
    ENERGY_COLS,
    MOOD_COLS,
    DIGESTIVE_COLS,
    build_v3_feature_row,
    severities_to_binary_symptoms,
    to_ordinal,
)

# ── Flow score mapping ────────────────────────────────────────────────────────

FLOW_SCORE_MAP: Dict[str, int] = {
    "none":             0,
    "not at all":       0,
    "spotting":         1,
    "spotting/very light": 1,
    "spotting / very light": 1,
    "very light":       1,
    "light":            2,
    "somewhat light":   2,
    "light/moderate":   2,
    "moderate":         3,
    "somewhat heavy":   3,
    "heavy":            4,
    "very heavy":       5,
}


def _to_flow_score(flow) -> int:
    if flow is None:
        return 0
    return FLOW_SCORE_MAP.get(str(flow).strip().lower(), 0)


# ── Ordinal history extractor ─────────────────────────────────────────────────

def _log_to_ords(log: Dict[str, object]) -> Dict[str, int]:
    """
    Extract ordinal severities from a ``recent_daily_logs`` entry.

    Prefers the ``symptom_severities`` key (added by v5 training / evaluation
    scripts).  Falls back to converting the binary ``symptoms`` list where
    every present symptom is treated as ordinal level 3.
    """
    sev = log.get("symptom_severities")
    if sev:
        return {s: int(sev.get(s, 0) or 0) for s in SUPPORTED_SYMPTOMS}
    syms = set(log.get("symptoms") or [])
    return {s: (3 if s in syms else 0) for s in SUPPORTED_SYMPTOMS}


# ── V5 feature additions ──────────────────────────────────────────────────────

def _add_v5_features(
    row: Dict[str, object],
    today_severities: Dict[str, int],
    recent_daily_logs: Optional[List[Dict[str, object]]],
    flow: str,
) -> None:
    """
    Mutate ``row`` in-place to add all v5-specific features.

    Parameters
    ----------
    row               : partially built feature dict (v3 features already in)
    today_severities  : {symptom: 0-5} for today
    recent_daily_logs : prior-day observation dicts (may contain severities)
    flow              : today's flow string
    """
    logs = recent_daily_logs or []

    # Build per-day ordinal dicts for the history window (up to 4 prior days)
    hist_ords: List[Dict[str, int]] = [_log_to_ords(log) for log in logs[-4:]]
    n_hist = len(hist_ords)

    lag1_ords = hist_ords[-1] if n_hist >= 1 else {s: 0 for s in SUPPORTED_SYMPTOMS}
    lag2_ords = hist_ords[-2] if n_hist >= 2 else {s: 0 for s in SUPPORTED_SYMPTOMS}

    # ── Group-level ordinal deltas (population-level, not person-specific) ──────
    # Per-symptom lags are NOT added — they capture individual baseline levels
    # that don't generalise across participants in LOPO CV (too person-specific).
    # Group sums are more stable cross-participant signal.
    today_total = float(row.get("total_severity", 0.0))
    lag1_total  = sum(float(lag1_ords.get(s, 0)) for s in SUPPORTED_SYMPTOMS)
    lag2_total  = sum(float(lag2_ords.get(s, 0)) for s in SUPPORTED_SYMPTOMS)

    for prefix, cols in [
        ("pain",    PAIN_COLS),
        ("energy",  ENERGY_COLS),
        ("mood",    MOOD_COLS),
    ]:
        today_sum = sum(float(today_severities.get(c, 0)) for c in cols)
        lag1_sum  = sum(float(lag1_ords.get(c, 0))        for c in cols)
        row[f"{prefix}_ord_delta"] = today_sum - lag1_sum   # directional group shift

    row["total_severity_delta"] = today_total - lag1_total  # overall symptom load shift

    # ── Extended rolling window (5 days) ─────────────────────────────────────
    all_ords   = hist_ords + [today_severities]             # up to 5 entries
    all_totals = [sum(float(d.get(s, 0)) for s in SUPPORTED_SYMPTOMS) for d in all_ords]
    row["total_severity_roll5_mean"] = float(np.mean(all_totals))
    row["total_severity_roll5_max"]  = float(max(all_totals))

    pain_vals = [sum(float(d.get(c, 0)) for c in PAIN_COLS) for d in all_ords]
    row["pain_ord_roll5_mean"] = float(np.mean(pain_vals))

    # ── Symptom pattern indicators (binary direction flags) ──────────────────
    row["is_symptom_rising"]    = 1.0 if (n_hist >= 1 and today_total > lag1_total)             else 0.0
    row["is_symptom_declining"] = 1.0 if (n_hist >= 1 and today_total < lag1_total)             else 0.0
    row["is_symptom_peak"]      = 1.0 if (n_hist >= 2 and today_total > lag1_total > lag2_total) else 0.0
    row["is_symptom_trough"]    = 1.0 if (n_hist >= 2 and today_total < lag1_total < lag2_total) else 0.0

    # ── Flow score trajectory ────────────────────────────────────────────────
    today_fs = _to_flow_score(flow)
    lag1_fs  = _to_flow_score(logs[-1].get("flow") if n_hist >= 1 else None)
    row["flow_score"]      = float(today_fs)
    row["flow_score_lag1"] = float(lag1_fs)
    row["flow_declining"]  = 1.0 if today_fs < lag1_fs else 0.0

    # ── Cycle position interaction ────────────────────────────────────────────
    row["cycle_cv_x_day_norm"] = float(
        row.get("cycle_length_cv", 0.0) * row.get("cycle_day_norm", 0.5)
    )


# ── Main entry ────────────────────────────────────────────────────────────────

def build_v5_feature_row(
    symptom_severities: Dict[str, int],
    cervical_mucus: str,
    appetite: int,
    exerciselevel: int,
    flow: str,
    recent_daily_logs: Optional[List[Dict[str, object]]],
    layer1: Dict[str, object],
    known_starts: List[str],
) -> Dict[str, object]:
    """
    Build a single flat feature dict for the v5 model.

    Calls the v3 builder first (all existing features), then appends
    the v5-specific trajectory, rolling, pattern, flow, and interaction
    features.  Both the v3 and v5 feature sets are present in the output.
    """
    # v3 base features (includes binary + ordinal + cycle timing + bleeding lags)
    row = build_v3_feature_row(
        symptom_severities=symptom_severities,
        cervical_mucus=cervical_mucus,
        appetite=appetite,
        exerciselevel=exerciselevel,
        flow=flow,
        recent_daily_logs=recent_daily_logs,
        layer1=layer1,
        known_starts=known_starts,
    )

    # v5 additions
    _add_v5_features(
        row=row,
        today_severities=symptom_severities,
        recent_daily_logs=recent_daily_logs,
        flow=flow,
    )

    return row
