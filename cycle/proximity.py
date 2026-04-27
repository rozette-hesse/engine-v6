"""
Period Proximity Model
======================
Predicts whether the user's current symptom pattern is consistent with the
pre-menstrual window (within 7 days of the next period) using symptom
features only — no cycle-day, no timing history.

Designed as a secondary signal for users with irregular cycles where
timing-based estimates are unreliable.

Accuracy (LOPO CV, honest): 55.0% balanced accuracy, 47.4% soon-recall.
Treat this as a soft signal, not a precise prediction.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR
from .layer2.base import _normalize_symptom_list
from .layer2.features_v3 import SUPPORTED_SYMPTOMS, to_ordinal


PROXIMITY_PIPELINE_FILE     = "proximity_pipeline.joblib"
PROXIMITY_FEATURE_COLS_FILE = "proximity_feature_columns.joblib"


def _load(key: str, path: Path):
    """Local artifact cache (separate from layer2 cache to avoid key conflicts)."""
    if key not in _PROXIMITY_CACHE:
        import joblib
        _PROXIMITY_CACHE[key] = joblib.load(path)
    return _PROXIMITY_CACHE[key]


_PROXIMITY_CACHE: Dict[str, object] = {}


def _resolve_severities(
    symptom_severities: Optional[Dict[str, int]],
    symptoms: Optional[List[str]],
) -> Dict[str, int]:
    """Accept either a severity dict or a legacy binary symptom list."""
    if symptom_severities:
        return {s: int(symptom_severities.get(s, 0) or 0) for s in SUPPORTED_SYMPTOMS}
    norm = set(_normalize_symptom_list(symptoms or []))
    return {s: (3 if s in norm else 0) for s in SUPPORTED_SYMPTOMS}


def get_proximity_output(
    symptoms: Optional[List[str]] = None,
    appetite: int = 0,
    exerciselevel: int = 0,
    *,
    symptom_severities: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """
    Run the period-proximity classifier on the current symptom state.

    Returns
    -------
    dict with keys:
      prediction   — "soon" | "not_soon"
      probability  — float, confidence that period is within 7 days
      signal       — "premenstrual_pattern" | "none"
      note         — human-readable explanation
    """
    pipeline     = _load("pipeline",     ARTIFACTS_DIR / PROXIMITY_PIPELINE_FILE)
    feature_cols = _load("feature_cols", ARTIFACTS_DIR / PROXIMITY_FEATURE_COLS_FILE)

    severities = _resolve_severities(symptom_severities, symptoms)

    feat: Dict[str, object] = {}
    for sym in SUPPORTED_SYMPTOMS:
        feat[f"{sym}_ord"]        = to_ordinal(severities.get(sym, 0))
    feat["appetite_ord"]          = to_ordinal(appetite)
    feat["exerciselevel_ord"]     = to_ordinal(exerciselevel)
    feat["total_severity"]        = sum(int(severities.get(s, 0) or 0) for s in SUPPORTED_SYMPTOMS)

    # No meaningful signals → skip model to avoid spurious results
    if feat["total_severity"] == 0 and int(appetite or 0) == 0:
        return {"prediction": "not_soon", "probability": 0.0, "signal": "none", "note": ""}

    X = pd.DataFrame([{col: feat.get(col, np.nan) for col in feature_cols}])

    proba    = pipeline.predict_proba(X)[0]
    classes  = list(pipeline.classes_)
    prob_map = dict(zip(classes, proba))

    soon_prob  = float(prob_map.get("soon", 0.0))
    # 0.65 threshold avoids low-symptom false positives
    prediction = "soon" if soon_prob >= 0.65 else "not_soon"

    if prediction == "soon" and soon_prob >= 0.80:
        signal = "premenstrual_pattern"
        note   = "Symptom pattern is consistent with pre-menstrual phase. Consider being prepared."
    elif prediction == "soon":
        signal = "premenstrual_pattern"
        note   = "Some symptoms associated with pre-menstrual phase detected."
    else:
        signal = "none"
        note   = ""

    return {
        "prediction":  prediction,
        "probability": round(soon_prob, 3),
        "signal":      signal,
        "note":        note,
    }
