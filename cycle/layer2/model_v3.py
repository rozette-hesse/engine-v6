"""
Layer 2 v3 — LightGBM with Ordinal Severity + Cycle Context
=============================================================
Loads the v3 LightGBM pipeline and builds the richer ordinal feature set
(including cycle-day features and L1 priors).  Output dict shape is
identical to model_v1 and model_v4 so the fusion layer can route
transparently based on MODEL_VERSION.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import (
    ARTIFACTS_DIR,
    LAYER2_V3_FEATURE_COLUMNS_FILE,
    LAYER2_V3_PIPELINE_FILE,
    NON_MENSTRUAL_PHASES,
    SUPPORTED_SYMPTOMS,
)
from .base import (
    _get_fertility_status,
    _get_signal_confidence,
    _normalize_symptom_list,
    load_artifact,
)
from .features_v3 import build_v3_feature_row


def _resolve_severities(
    symptom_severities: Optional[Dict[str, int]],
    symptoms: Optional[List[str]],
) -> Dict[str, int]:
    """Accept either a severity dict or a legacy binary list (treated as severity 3)."""
    if symptom_severities:
        return {s: int(symptom_severities.get(s, 0) or 0) for s in SUPPORTED_SYMPTOMS}
    norm = set(_normalize_symptom_list(symptoms))
    return {s: (3 if s in norm else 0) for s in SUPPORTED_SYMPTOMS}


def _build_v3_frame(
    severities: Dict[str, int],
    cervical_mucus: str,
    appetite: int,
    exerciselevel: int,
    flow: str,
    recent_daily_logs: Optional[List[Dict[str, object]]],
    layer1: Dict[str, object],
    known_starts: List[str],
) -> pd.DataFrame:
    feature_cols = load_artifact("v3_feature_cols", ARTIFACTS_DIR / LAYER2_V3_FEATURE_COLUMNS_FILE)
    final_row    = build_v3_feature_row(
        symptom_severities=severities,
        cervical_mucus=cervical_mucus,
        appetite=appetite,
        exerciselevel=exerciselevel,
        flow=flow,
        recent_daily_logs=recent_daily_logs,
        layer1=layer1,
        known_starts=known_starts,
    )
    X = pd.DataFrame([{col: final_row.get(col, np.nan) for col in feature_cols}])
    for c in X.columns:
        if X[c].dtype == "object" and X[c].isna().all():
            X[c] = "unknown"
    return X


def get_layer2_v3_output(
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
    """Run v3 inference and return a standardized output dict."""
    pipeline    = load_artifact("v3_pipeline", ARTIFACTS_DIR / LAYER2_V3_PIPELINE_FILE)
    # v3 was fit with raw string labels — pipeline.classes_ holds them directly
    class_names = list(pipeline.classes_)

    severities   = _resolve_severities(symptom_severities, symptoms)
    layer1       = layer1 or {}
    known_starts = known_starts or []

    X = _build_v3_frame(
        severities=severities,
        cervical_mucus=cervical_mucus,
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
            "v3 ordinal-severity model used current symptom load and cycle position to refine the phase estimate."
        ],
        "features_used": X.to_dict(orient="records")[0],
        "model_version": "v3",
    }
