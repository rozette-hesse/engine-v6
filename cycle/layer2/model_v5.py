"""
Layer 2 v5 — Trajectory-Aware LightGBM
========================================
Extends v4 with ordinal severity trajectory features, extended 5-day rolling
windows, flow score history, symptom pattern indicators, and cycle-position
interaction features.  Mucus is still excluded from model training (same
as v4 — leakage prevention).  User-reported mucus applies the same
inference-time bonus in fusion.py.

Output dict shape is identical to v1, v3, and v4 so the fusion layer
requires no changes other than routing the "v5" version string.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import (
    ARTIFACTS_DIR,
    LAYER2_V5_FEATURE_COLUMNS_FILE,
    LAYER2_V5_PIPELINE_FILE,
    NON_MENSTRUAL_PHASES,
    SUPPORTED_SYMPTOMS,
)
from .base import (
    _get_fertility_status,
    _get_signal_confidence,
    _normalize_symptom_list,
    load_artifact,
)
from .features_v5 import build_v5_feature_row


def _resolve_severities(
    symptom_severities: Optional[Dict[str, int]],
    symptoms: Optional[List[str]],
) -> Dict[str, int]:
    if symptom_severities:
        return {s: int(symptom_severities.get(s, 0) or 0) for s in SUPPORTED_SYMPTOMS}
    norm = set(_normalize_symptom_list(symptoms))
    return {s: (3 if s in norm else 0) for s in SUPPORTED_SYMPTOMS}


def _build_v5_frame(
    severities: Dict[str, int],
    appetite: int,
    exerciselevel: int,
    flow: str,
    recent_daily_logs: Optional[List[Dict[str, object]]],
    layer1: Dict[str, object],
    known_starts: List[str],
) -> pd.DataFrame:
    feature_cols = load_artifact(
        "v5_feature_cols", ARTIFACTS_DIR / LAYER2_V5_FEATURE_COLUMNS_FILE
    )
    full_row = build_v5_feature_row(
        symptom_severities=severities,
        cervical_mucus="unknown",  # mucus excluded from model (label-leakage guard)
        appetite=appetite,
        exerciselevel=exerciselevel,
        flow=flow,
        recent_daily_logs=recent_daily_logs,
        layer1=layer1,
        known_starts=known_starts,
    )
    X = pd.DataFrame([{col: full_row.get(col, np.nan) for col in feature_cols}])
    for c in X.columns:
        if X[c].dtype == "object" and X[c].isna().all():
            X[c] = "unknown"
    return X


def get_layer2_v5_output(
    symptoms: Optional[List[str]] = None,
    cervical_mucus: str = "unknown",
    appetite: int = 0,
    exerciselevel: int = 0,
    recent_daily_logs: Optional[List[Dict[str, object]]] = None,
    flow: str = "none",
    *,
    symptom_severities: Optional[Dict[str, int]] = None,
    layer1: Optional[Dict[str, object]] = None,
    known_starts: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Run v5 inference and return a standardized output dict."""
    pipeline    = load_artifact("v5_pipeline", ARTIFACTS_DIR / LAYER2_V5_PIPELINE_FILE)
    class_names = list(pipeline.classes_)

    severities   = _resolve_severities(symptom_severities, symptoms)
    layer1       = layer1 or {}
    known_starts = known_starts or []

    X = _build_v5_frame(
        severities=severities,
        appetite=appetite,
        exerciselevel=exerciselevel,
        flow=flow,
        recent_daily_logs=recent_daily_logs,
        layer1=layer1,
        known_starts=known_starts,
    )

    probs       = pipeline.predict_proba(X)[0]
    phase_probs = {phase: float(prob) for phase, prob in zip(class_names, probs)}
    for phase in NON_MENSTRUAL_PHASES:
        phase_probs.setdefault(phase, 0.0)
    total       = sum(phase_probs.values()) or 1.0
    phase_probs = {k: v / total for k, v in phase_probs.items()}

    sorted_items = sorted(phase_probs.items(), key=lambda x: x[1], reverse=True)
    top_phase    = sorted_items[0][0]
    top_prob     = sorted_items[0][1]
    second_prob  = sorted_items[1][1] if len(sorted_items) > 1 else 0.0

    symptom_count = sum(1 for v in severities.values() if v >= 3)

    return {
        "phase_probs":       phase_probs,
        "top_phase":         top_phase,
        "top_prob":          float(top_prob),
        "second_prob":       float(second_prob),
        "prob_gap":          float(top_prob - second_prob),
        "fertility_status":  _get_fertility_status(phase_probs, cervical_mucus, symptom_count),
        "signal_confidence": _get_signal_confidence(phase_probs, symptom_count, cervical_mucus),
        "explanations": [
            "v5 trajectory-aware model used symptom trends and cycle position "
            "to refine the phase estimate."
        ],
        "features_used": X.to_dict(orient="records")[0],
        "model_version": "v5",
    }
