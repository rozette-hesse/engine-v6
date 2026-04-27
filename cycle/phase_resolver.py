"""
Phase Resolver
==============
Bridges the 4-phase cycle engine output to the 5-phase code used by the
recommendation engine.

Cycle engine phases : Menstrual, Follicular, Fertility, Luteal
Recommender phases  : MEN, FOL, OVU, LUT, LL

Mapping
-------
  Menstrual   → MEN
  Follicular  → FOL
  Fertility   → OVU
  Luteal      → LUT  (early / mid luteal)
              → LL   (late luteal: ≤ LL_THRESHOLD_DAYS before predicted period)
"""

from typing import Dict, Optional


_PHASE_MAP = {
    "Menstrual":  "MEN",
    "Follicular": "FOL",
    "Fertility":  "OVU",
}

# Days before predicted period that qualify as Late Luteal
LL_THRESHOLD_DAYS = 5


def resolve_phase(
    final_phase: str,
    cycle_day: Optional[int],
    estimated_cycle_length: Optional[float],
) -> str:
    """
    Convert a 4-phase label into the 5-phase code the recommender expects.

    Luteal is split into LUT (early/mid) and LL (late luteal, ≤ 5 days
    before the predicted next period).
    """
    if final_phase in _PHASE_MAP:
        return _PHASE_MAP[final_phase]

    # Luteal: decide between LUT and LL based on cycle position
    if final_phase == "Luteal" and cycle_day and estimated_cycle_length:
        days_remaining = round(estimated_cycle_length) - cycle_day
        if days_remaining <= LL_THRESHOLD_DAYS:
            return "LL"

    return "LUT"


def resolve_phase_from_fusion(fusion_output: Dict) -> str:
    """Convenience wrapper: extract phase from a full fusion output dict."""
    layer1 = fusion_output.get("layer1") or {}
    return resolve_phase(
        final_phase=fusion_output["final_phase"],
        cycle_day=layer1.get("cycle_day"),
        estimated_cycle_length=layer1.get("estimated_cycle_length"),
    )
