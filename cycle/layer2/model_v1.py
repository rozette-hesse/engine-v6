"""
Layer 2 v1 (vnext) — Legacy RandomForest Model
================================================
DEPRECATED.  This model has no recoverable training script and its
validation metrics were measured on training data.

Use model_v3 or model_v4 for new work.  This module is retained only for
backwards-compatibility with environments that have not yet migrated.
"""

from typing import Dict, List, Optional

import numpy as np

from ..config import (
    ARTIFACTS_DIR,
    LAYER2_LABEL_ENCODER_FILE,
    LAYER2_PIPELINE_FILE,
    NON_MENSTRUAL_PHASES,
)
from .base import (
    _build_explanations,
    _get_fertility_status,
    _get_signal_confidence,
    _make_feature_frame,
    _normalize_symptom_list,
    load_artifact,
)


def get_layer2_output(
    symptoms: Optional[List[str]],
    cervical_mucus: str = "unknown",
    appetite: int = 0,
    exerciselevel: int = 0,
    recent_daily_logs: Optional[List[Dict[str, object]]] = None,
    flow: str = "none",
) -> Dict[str, object]:
    """
    Run v1 (vnext) inference.  Produces the same output dict shape as v3/v4
    so the fusion layer can route to any version transparently.

    Parameters are identical to the v3/v4 signature except that
    symptom_severities, layer1, and known_starts are not used.
    """
    pipeline      = load_artifact("v1_pipeline",      ARTIFACTS_DIR / LAYER2_PIPELINE_FILE)
    label_encoder = load_artifact("v1_label_encoder", ARTIFACTS_DIR / LAYER2_LABEL_ENCODER_FILE)

    X = _make_feature_frame(symptoms, cervical_mucus, appetite, exerciselevel, recent_daily_logs, flow)

    probs          = pipeline.predict_proba(X)[0]
    classes_enc    = pipeline.classes_
    class_names    = label_encoder.inverse_transform(classes_enc)
    phase_probs    = {phase: float(prob) for phase, prob in zip(class_names, probs)}

    for phase in NON_MENSTRUAL_PHASES:
        phase_probs.setdefault(phase, 0.0)
    total       = sum(phase_probs.values()) or 1.0
    phase_probs = {k: v / total for k, v in phase_probs.items()}

    top_phase      = max(phase_probs, key=phase_probs.get)
    symptom_count  = len(_normalize_symptom_list(symptoms))
    sorted_items   = sorted(phase_probs.items(), key=lambda x: x[1], reverse=True)
    top_prob       = sorted_items[0][1]
    second_prob    = sorted_items[1][1] if len(sorted_items) > 1 else 0.0

    return {
        "phase_probs":       phase_probs,
        "top_phase":         top_phase,
        "top_prob":          float(top_prob),
        "second_prob":       float(second_prob),
        "prob_gap":          float(top_prob - second_prob),
        "fertility_status":  _get_fertility_status(phase_probs, cervical_mucus, symptom_count),
        "signal_confidence": _get_signal_confidence(phase_probs, symptom_count, cervical_mucus),
        "explanations":      _build_explanations(symptoms, cervical_mucus, top_phase, recent_daily_logs),
        "features_used":     X.to_dict(orient="records")[0],
        "model_version":     "v1",
    }
