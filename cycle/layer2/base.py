"""
Layer 2 shared helpers and model cache.

This module is the foundation for all three Layer 2 model variants (v1, v3, v4).
It provides:
  - Symptom normalisation (alias resolution, binary list → symptom set)
  - Cervical mucus normalisation and fertility scoring
  - Feature row construction (today's row + rolling 3-day history)
  - Signal quality assessment (fertility status, signal confidence)
  - Explanation generation
  - A shared artifact cache so each joblib file is loaded exactly once
    regardless of which model variant requested it first.
"""

from typing import Dict, List, Optional
import warnings

import joblib
import numpy as np
import pandas as pd

from ..config import (
    ARTIFACTS_DIR,
    LAYER2_FEATURE_COLUMNS_FILE,
    LAYER2_LABEL_ENCODER_FILE,
    LAYER2_PIPELINE_FILE,
    MODEL_VERSION,
    NON_MENSTRUAL_PHASES,
    SUPPORTED_SYMPTOMS,
)


# ── v1 (vnext) deprecation notice ────────────────────────────────────────────

if MODEL_VERSION not in ("v2", "v3", "v4"):
    warnings.warn(
        "Layer 2 vnext (v1) is deprecated and has no recoverable training script. "
        "Its validation metrics were measured on training data. "
        "Set INBALANCE_MODEL_VERSION=v3 or v4 to use a supported model. "
        "vnext will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )


# ── Symptom normalisation ─────────────────────────────────────────────────────

# Map common spelling variants to canonical symptom names
SYMPTOM_ALIASES: Dict[str, str] = {
    "headache": "headaches", "headaches": "headaches",
    "cramp": "cramps",       "cramps": "cramps",
    "sorebreast": "sorebreasts", "sorebreasts": "sorebreasts",
    "fatigue": "fatigue",
    "sleepissue": "sleepissue", "sleepissues": "sleepissue", "sleep_issue": "sleepissue",
    "moodswing": "moodswing",   "moodswings": "moodswing",
    "stress": "stress",
    "foodcraving": "foodcravings", "foodcravings": "foodcravings",
    "indigestion": "indigestion",
    "bloating": "bloating",
}

# Cervical mucus → fertility score (0.0 = not fertile, 1.0 = peak fertile)
MUCUS_FERTILITY_MAP: Dict[str, float] = {
    "unknown": 0.0,
    "dry":     0.0,
    "sticky":  0.25,
    "creamy":  0.50,
    "watery":  0.85,
    "eggwhite": 1.00,
}


def _safe_int(value, default: int = 0) -> int:
    try:
        return default if value is None else int(value)
    except Exception:
        return default


def _normalize_symptom_name(symptom: str) -> Optional[str]:
    if symptom is None:
        return None
    key = str(symptom).strip().lower().replace(" ", "").replace("-", "").replace("/", "")
    return SYMPTOM_ALIASES.get(key)


def _normalize_symptom_list(symptoms: Optional[List[str]]) -> List[str]:
    """Resolve a list of symptom strings to canonical names, deduplicated."""
    out = []
    for s in symptoms or []:
        norm = _normalize_symptom_name(s)
        if norm and norm in SUPPORTED_SYMPTOMS and norm not in out:
            out.append(norm)
    return out


def _normalize_mucus(mucus: Optional[str]) -> str:
    if mucus is None:
        return "unknown"
    m = str(mucus).strip().lower()
    return m if m in MUCUS_FERTILITY_MAP else "unknown"


# ── Feature row construction ──────────────────────────────────────────────────

def _build_today_row(
    symptoms: Optional[List[str]],
    cervical_mucus: str = "unknown",
    appetite: int = 0,
    exerciselevel: int = 0,
    flow: str = "none",
) -> Dict[str, object]:
    """
    Build a single feature dict for today's observations.

    Populates binary symptom flags, group aggregate features (mean/max across
    pain, energy, mood, digestive groups), mucus-derived features, and
    bleeding_present from flow input.
    """
    symptom_set = set(_normalize_symptom_list(symptoms))
    mucus_type  = _normalize_mucus(cervical_mucus)
    row: Dict[str, object] = {}

    for sym in SUPPORTED_SYMPTOMS:
        row[sym]              = 1.0 if sym in symptom_set else 0.0
        row[f"{sym}_logged"]  = 1

    pain_cols      = ["headaches", "cramps", "sorebreasts"]
    energy_cols    = ["fatigue", "sleepissue"]
    mood_cols      = ["moodswing", "stress"]
    digestive_cols = ["foodcravings", "indigestion", "bloating"]

    def _add_group(group_name: str, cols: List[str]) -> None:
        values = [float(row[c]) for c in cols]
        row[f"{group_name}_mean"]          = float(np.mean(values)) if values else 0.0
        row[f"{group_name}_max"]           = float(np.max(values))  if values else 0.0
        row[f"{group_name}_logged_count"]  = len(cols)
        row[f"{group_name}_missing_frac"]  = 0.0

    _add_group("pain",      pain_cols)
    _add_group("energy",    energy_cols)
    _add_group("mood",      mood_cols)
    _add_group("digestive", digestive_cols)

    row["num_symptoms_logged"]  = len(SUPPORTED_SYMPTOMS)
    row["symptom_completeness"] = 1.0

    row["mucus_logged"]           = 0 if mucus_type == "unknown" else 1
    row["mucus_score_logged"]     = 0 if mucus_type == "unknown" else 1
    row["mucus_fertility_score"]  = float(MUCUS_FERTILITY_MAP[mucus_type])

    row["appetite"]      = _safe_int(appetite, 0)
    row["exerciselevel"] = _safe_int(exerciselevel, 0)
    row["mucus_type"]    = mucus_type

    flow_val              = str(flow or "none").strip().lower()
    row["bleeding_present"] = 1.0 if flow_val not in {"none", "not at all", "", "unknown"} else 0.0

    return row


def _build_recent_rows(
    recent_daily_logs: Optional[List[Dict[str, object]]],
    today_row: Dict[str, object],
) -> List[Dict[str, object]]:
    """Build a rolling window of up to 3 feature rows (history + today)."""
    rows = []
    for item in recent_daily_logs or []:
        rows.append(_build_today_row(
            symptoms=item.get("symptoms", []),
            cervical_mucus=item.get("cervical_mucus", "unknown"),
            appetite=item.get("appetite", 0),
            exerciselevel=item.get("exerciselevel", 0),
            flow=item.get("flow", "none"),
        ))
    rows.append(today_row)
    return rows[-3:]


def _apply_history_features(
    last_rows: List[Dict[str, object]],
    today_row: Dict[str, object],
) -> Dict[str, object]:
    """
    Augment today_row with lag, rolling, and trend features derived from
    the 3-day history window.  These features are what the v1/v2 model
    was trained on.
    """
    out  = dict(today_row)
    rows = last_rows[-3:]
    n    = len(rows)

    history_base_cols = [
        "pain_mean", "pain_max", "energy_mean", "energy_max",
        "mood_mean", "mood_max", "digestive_mean", "digestive_max",
        "mucus_fertility_score", "num_symptoms_logged",
        "symptom_completeness", "bleeding_present",
    ]

    def val(r_idx: int, col: str) -> float:
        if r_idx < 0 or r_idx >= n:
            return 0.0
        try:
            v = rows[r_idx].get(col, 0.0)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 0.0
            return float(v)
        except Exception:
            return 0.0

    for col in history_base_cols:
        current = val(n - 1, col)
        lag1    = val(n - 2, col) if n >= 2 else 0.0
        lag2    = val(n - 3, col) if n >= 3 else 0.0
        series  = [val(i, col) for i in range(n)]

        out[f"{col}_lag1"]       = lag1
        out[f"{col}_lag2"]       = lag2
        out[f"{col}_roll3_mean"] = float(np.mean(series)) if series else current
        out[f"{col}_roll3_max"]  = float(np.max(series))  if series else current
        out[f"{col}_trend2"]     = float(current - lag2)
        out[f"{col}_persist3"]   = float(sum(1 for x in series if x > 0))

    return out


def _make_feature_frame(
    symptoms: Optional[List[str]],
    cervical_mucus: str = "unknown",
    appetite: int = 0,
    exerciselevel: int = 0,
    recent_daily_logs: Optional[List[Dict[str, object]]] = None,
    flow: str = "none",
) -> pd.DataFrame:
    """Build the final v1/v2-style feature DataFrame for a single inference row."""
    today_row    = _build_today_row(symptoms, cervical_mucus, appetite, exerciselevel, flow)
    history_rows = _build_recent_rows(recent_daily_logs, today_row)
    final_row    = _apply_history_features(history_rows, today_row)

    feature_cols = load_artifact("feature_cols", ARTIFACTS_DIR / LAYER2_FEATURE_COLUMNS_FILE)
    X = pd.DataFrame([{col: final_row.get(col, np.nan) for col in feature_cols}])
    for c in X.columns:
        if X[c].dtype == "object" and X[c].isna().all():
            X[c] = "unknown"
    return X


# ── Signal quality ────────────────────────────────────────────────────────────

def _get_signal_confidence(
    phase_probs: Dict[str, float],
    symptom_count: int,
    cervical_mucus: str,
) -> str:
    """Classify how strongly the symptom signals point toward a phase."""
    sorted_probs = sorted(phase_probs.values(), reverse=True)
    top_prob     = sorted_probs[0] if sorted_probs else 0.0
    second_prob  = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    gap          = top_prob - second_prob
    mucus        = _normalize_mucus(cervical_mucus)

    if gap >= 0.30 and (symptom_count >= 2 or mucus in {"watery", "eggwhite"}):
        return "high"
    if gap >= 0.15 and (symptom_count >= 1 or mucus != "unknown"):
        return "medium"
    return "low"


def _get_fertility_status(
    phase_probs: Dict[str, float],
    cervical_mucus: str,
    symptom_count: int,
) -> str:
    """Classify the current fertility window: Red Day, Light Red Day, Green Day."""
    fertility_prob = phase_probs.get("Fertility", 0.0)
    mucus          = _normalize_mucus(cervical_mucus)

    if mucus in {"watery", "eggwhite"} and fertility_prob >= 0.40:
        return "Red Day"
    if fertility_prob >= 0.60:
        return "Red Day"
    if fertility_prob >= 0.35 or mucus in {"creamy", "watery"}:
        return "Light Red Day"
    if symptom_count == 0 and mucus == "unknown":
        return "Need More Data"
    return "Green Day"


def _build_explanations(
    symptoms: Optional[List[str]],
    cervical_mucus: str,
    top_phase: str,
    recent_daily_logs: Optional[List[Dict[str, object]]] = None,
) -> List[str]:
    """Generate up to 3 plain-language explanation strings for the phase prediction."""
    symptom_set  = set(_normalize_symptom_list(symptoms))
    mucus        = _normalize_mucus(cervical_mucus)
    explanations = []

    if mucus in {"watery", "eggwhite"}:
        explanations.append("Fertile-type cervical mucus increased fertility likelihood.")
    elif mucus == "creamy":
        explanations.append("Creamy cervical mucus supported a possible fertility transition.")
    elif mucus in {"dry", "sticky"}:
        explanations.append("Dry or sticky mucus lowered fertile-window likelihood.")

    if top_phase == "Luteal" and {"sorebreasts", "foodcravings", "bloating"} & symptom_set:
        explanations.append("Breast tenderness, cravings, or bloating supported a luteal-like pattern.")

    if top_phase == "Follicular" and len(symptom_set) <= 2 and mucus in {"unknown", "sticky", "creamy", "dry"}:
        explanations.append("Lower or less specific symptom burden fit better with a follicular-like pattern.")

    if top_phase == "Fertility" and mucus in {"watery", "eggwhite", "creamy"}:
        explanations.append("Body signals matched a more fertile phase pattern.")

    if recent_daily_logs:
        explanations.append("Recent 3-day symptom history was used to stabilize the phase estimate.")

    if not explanations:
        explanations.append("Current symptom pattern was used to refine today's non-menstrual phase estimate.")

    return explanations[:3]


# ── Shared artifact cache ─────────────────────────────────────────────────────
# A single cache shared across all model variants so each .joblib file is
# loaded at most once per process lifetime.

_ARTIFACT_CACHE: Dict[str, object] = {}


def load_artifact(key: str, path) -> object:
    """Load a joblib artifact, using a process-level cache to avoid repeated disk reads."""
    if key not in _ARTIFACT_CACHE:
        _ARTIFACT_CACHE[key] = joblib.load(path)
    return _ARTIFACT_CACHE[key]
