"""
Recommendation engine package.

Public interface
----------------
  recommend(inputs, top_n=3, phase_mode="phase_aware")
      → dict[domain][category][items]
  suggest_phase_mode(fusion) → "phase_aware" | "phase_blind"

  build_concept_activations, spread, check_guards are also exported
  for testing and audit tooling. PHASE_AWARE / PHASE_BLIND are string
  constants for the phase_mode argument.
"""

from .engine import (
    recommend,
    suggest_phase_mode,
    build_concept_activations,
    spread,
    check_guards,
    PHASE_AWARE,
    PHASE_BLIND,
)

__all__ = [
    "recommend",
    "suggest_phase_mode",
    "build_concept_activations",
    "spread",
    "check_guards",
    "PHASE_AWARE",
    "PHASE_BLIND",
]
