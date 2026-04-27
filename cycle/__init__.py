"""
cycle — Menstrual Cycle Prediction Engine
==========================================
Public interface:

  get_fused_output(period_starts, symptoms, ...) → full multi-layer fusion result
  get_proximity_output(symptoms, ...)            → period proximity prediction
  resolve_phase_from_fusion(fusion_output)       → 5-phase recommender code
  compute_personal_stats(history)               → per-phase symptom baselines
"""

from .fusion import get_fused_output
from .proximity import get_proximity_output
from .phase_resolver import resolve_phase_from_fusion
from .personalization import compute_personal_stats

__all__ = [
    "get_fused_output",
    "get_proximity_output",
    "resolve_phase_from_fusion",
    "compute_personal_stats",
]
