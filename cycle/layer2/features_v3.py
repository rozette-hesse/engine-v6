"""
Layer 2 v3/v4 Feature Builder
==============================
Builds the richer feature set used by the v3 and v4 LightGBM models.

Differences vs the v1/v2 feature set
-------------------------------------
1. Ordinal symptom severity preserved (0-5 scale) in addition to the legacy
   binary columns.  Both forms are emitted for backward compatibility.
2. Severity-weighted group aggregates: pain_ord_sum/max, energy_ord_sum/max, etc.
3. Richer cycle-timing features: cycle_day_norm, days_from_ovulation,
   is_past_ovulation, cyclical sin/cos encoding, days_until_next_period_norm.
4. Layer-1 prior probabilities injected as features (l1_prior_follicular, etc.)
   so the model can exploit cycle-day evidence learned during training.
5. Bleeding history: bleeding_lag1, bleeding_lag2, bleeding_streak3.

Entry point
-----------
  build_v3_feature_row(symptom_severities, cervical_mucus, appetite,
                       exerciselevel, flow, recent_daily_logs,
                       layer1, known_starts) → dict
"""

import math
from typing import Dict, List, Optional

import numpy as np

from ..config import SUPPORTED_SYMPTOMS
from ..layer1_period import compute_cycle_lengths
from .base import (
    MUCUS_FERTILITY_MAP,
    _apply_history_features,
    _build_recent_rows,
    _build_today_row,
    _normalize_mucus,
)


# ── Ordinal helpers ───────────────────────────────────────────────────────────

ORDINAL_INT: Dict[str, int] = {
    "Not at all":      0,
    "Very Low":        1,
    "Very Low/Little": 1,
    "Low":             2,
    "Moderate":        3,
    "High":            4,
    "Very High":       5,
}


def to_ordinal(value) -> int:
    """Coerce any cell value into a 0..5 integer severity."""
    if value is None:
        return 0
    if isinstance(value, float) and np.isnan(value):
        return 0
    s = str(value).strip()
    if s in ORDINAL_INT:
        return ORDINAL_INT[s]
    try:
        return max(0, min(5, int(float(s))))
    except (ValueError, TypeError):
        return 0


def severities_to_binary_symptoms(severities: Dict[str, int], threshold: int = 3) -> List[str]:
    """Convert a severity dict to a binary symptom list (present = severity >= threshold)."""
    return [s for s in SUPPORTED_SYMPTOMS if int(severities.get(s, 0)) >= threshold]


# ── Feature group definitions ─────────────────────────────────────────────────

PAIN_COLS      = ["headaches", "cramps", "sorebreasts"]
ENERGY_COLS    = ["fatigue", "sleepissue"]
MOOD_COLS      = ["moodswing", "stress"]
DIGESTIVE_COLS = ["foodcravings", "indigestion", "bloating"]


def _add_ordinal_features(row: Dict[str, object], severities: Dict[str, int]) -> None:
    """Mutate ``row`` to add ordinal symptom features and severity-weighted group stats."""
    total = 0.0
    for sym in SUPPORTED_SYMPTOMS:
        v = int(severities.get(sym, 0))
        row[f"{sym}_ord"] = float(v)
        total += v
    row["total_severity"] = total

    def group(prefix: str, cols: List[str]) -> None:
        vals = [float(severities.get(c, 0)) for c in cols]
        row[f"{prefix}_ord_sum"]  = float(sum(vals))
        row[f"{prefix}_ord_max"]  = float(max(vals)) if vals else 0.0
        row[f"{prefix}_ord_mean"] = float(np.mean(vals)) if vals else 0.0

    group("pain",      PAIN_COLS)
    group("energy",    ENERGY_COLS)
    group("mood",      MOOD_COLS)
    group("digestive", DIGESTIVE_COLS)


def _add_cycle_features(
    row: Dict[str, object],
    layer1: Dict[str, object],
    known_starts: List[str],
) -> None:
    """Mutate ``row`` to add cycle-position features and L1 prior probabilities."""
    cycle_day = layer1.get("cycle_day")
    avg_len   = layer1.get("estimated_cycle_length")
    ovul_day  = layer1.get("possible_ovulation_day")

    if cycle_day is not None and avg_len and avg_len > 0:
        row["cycle_day_norm"]              = float(cycle_day) / float(avg_len)
        row["days_from_ovulation"]         = float(cycle_day - ovul_day) if ovul_day else 0.0
        theta                              = 2.0 * math.pi * (float(cycle_day) / float(avg_len))
        row["cycle_day_sin"]               = math.sin(theta)
        row["cycle_day_cos"]               = math.cos(theta)
        row["days_until_next_period_norm"] = max(0.0, 1.0 - float(cycle_day) / float(avg_len))
    else:
        row["cycle_day_norm"]              = 0.5
        row["days_from_ovulation"]         = 0.0
        row["cycle_day_sin"]               = 0.0
        row["cycle_day_cos"]               = 1.0
        row["days_until_next_period_norm"] = 0.5

    row["is_past_ovulation"] = (
        1.0 if (cycle_day and ovul_day and cycle_day > ovul_day) else 0.0
    )

    cycle_lengths = compute_cycle_lengths(known_starts) if known_starts else []
    if len(cycle_lengths) >= 2:
        mean_len              = float(np.mean(cycle_lengths))
        std_len               = float(np.std(cycle_lengths))
        row["cycle_length_cv"] = std_len / mean_len if mean_len > 0 else 0.0
    else:
        row["cycle_length_cv"] = 0.0
    row["n_periods_logged"] = float(len(known_starts) if known_starts else 0)

    # Layer-1 priors as model features (3-class non-menstrual, renormalized)
    l1_probs = layer1.get("phase_probs") or {}
    l1_fol   = float(l1_probs.get("Follicular", 0.0) + l1_probs.get("Menstrual", 0.0))
    l1_fer   = float(l1_probs.get("Fertility",  0.0))
    l1_lut   = float(l1_probs.get("Luteal",     0.0))
    total    = l1_fol + l1_fer + l1_lut
    if total > 0:
        l1_fol, l1_fer, l1_lut = l1_fol / total, l1_fer / total, l1_lut / total
    row["l1_prior_follicular"] = l1_fol
    row["l1_prior_fertility"]  = l1_fer
    row["l1_prior_luteal"]     = l1_lut

    fc = str(layer1.get("forecast_confidence", "low")).lower()
    row["l1_forecast_confidence_high"]   = 1.0 if fc == "high"   else 0.0
    row["l1_forecast_confidence_medium"] = 1.0 if fc == "medium" else 0.0


def _add_bleeding_history(
    row: Dict[str, object],
    history_rows: List[Dict[str, object]],
) -> None:
    """Mutate ``row`` to add lag and streak features for bleeding_present."""
    rows = history_rows[-3:]
    n    = len(rows)

    def b(idx: int) -> float:
        if idx < 0 or idx >= n:
            return 0.0
        try:
            return float(rows[idx].get("bleeding_present", 0.0))
        except Exception:
            return 0.0

    today_b              = b(n - 1)
    row["bleeding_lag1"]    = b(n - 2)
    row["bleeding_lag2"]    = b(n - 3)
    row["bleeding_streak3"] = today_b + b(n - 2) + b(n - 3)


# ── Main entry ────────────────────────────────────────────────────────────────

def build_v3_feature_row(
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
    Build a single flat feature dict for the v3/v4 model.

    ``symptom_severities`` maps symptom name → 0..5 severity.
    Missing keys default to 0. Both binary (legacy) and ordinal columns are
    populated so the v3/v4 pipeline can use richer signal while preserving
    every column the v2 history pipeline already engineered.
    """
    binary_symptoms = severities_to_binary_symptoms(symptom_severities)

    today_row    = _build_today_row(binary_symptoms, cervical_mucus, appetite, exerciselevel, flow)
    history_rows = _build_recent_rows(recent_daily_logs, today_row)
    final_row    = _apply_history_features(history_rows, today_row)

    _add_ordinal_features(final_row, symptom_severities)
    _add_cycle_features(final_row, layer1, known_starts)
    _add_bleeding_history(final_row, history_rows)

    final_row.setdefault(
        "mucus_fertility_score",
        float(MUCUS_FERTILITY_MAP.get(_normalize_mucus(cervical_mucus), 0.0)),
    )

    return final_row
