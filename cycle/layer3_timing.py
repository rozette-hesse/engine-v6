"""
Layer 3 — Timing Narrative
===========================
Compares Layer 1 (timing-based phase) with Layer 2 (symptom-based phase)
and produces a human-readable status and note explaining any divergence.

This layer does not change the final phase decision — it only annotates it.
"""

from typing import Dict


PHASE_INDEX = {"Follicular": 0, "Fertility": 1, "Luteal": 2}


def _phase_distance(a: str, b: str) -> int:
    """Steps apart between two non-menstrual phases in the ordered sequence."""
    return abs(PHASE_INDEX[a] - PHASE_INDEX[b])


def get_timing_status(layer1: Dict, layer2: Dict, period_start_logged: bool = False) -> str:
    """Classify the relationship between history-based and symptom-based phase estimates."""
    if period_start_logged:
        return "Period started"

    layer1_phase = layer1.get("top_phase", max(layer1["phase_probs"], key=layer1["phase_probs"].get))
    layer2_phase = layer2["top_phase"]

    # Treat a Menstrual L1 reading as Luteal for distance comparison
    if layer1_phase == "Menstrual":
        layer1_phase = "Luteal"

    if layer1_phase == layer2_phase:
        return "On track"

    dist = _phase_distance(layer1_phase, layer2_phase)

    if dist == 1:
        if layer1_phase == "Follicular" and layer2_phase == "Fertility":
            return "Approaching ovulation earlier"
        if layer1_phase == "Fertility" and layer2_phase == "Follicular":
            return "Approaching ovulation later"
        if layer1_phase == "Fertility" and layer2_phase == "Luteal":
            return "Post-ovulation"
        if layer1_phase == "Luteal" and layer2_phase == "Fertility":
            return "Fertility signal stronger than timing"
        if layer1_phase == "Luteal" and layer2_phase == "Follicular":
            return "Timing uncertain this cycle"

    return "Timing uncertain this cycle"


def build_timing_note(
    layer1: Dict,
    layer2: Dict,
    timing_status: str,
    period_start_logged: bool = False,
) -> str:
    """Generate a plain-language explanation of the timing status."""
    if period_start_logged:
        return "You logged a period start today, so the app resets the cycle and marks today as menstrual."

    layer1_phase = layer1.get("top_phase", max(layer1["phase_probs"], key=layer1["phase_probs"].get))
    if layer1_phase == "Menstrual":
        layer1_phase = "Luteal"

    layer2_phase = layer2["top_phase"]

    if timing_status == "On track":
        return f"History and body signals both support a {layer2_phase.lower()} pattern today."
    if timing_status == "Approaching ovulation earlier":
        return (
            f"Timing still leans {layer1_phase.lower()}, but symptoms suggest movement "
            "toward fertility sooner than expected."
        )
    if timing_status == "Approaching ovulation later":
        return f"Timing suggested fertility, but current symptoms still look more {layer2_phase.lower()}."
    if timing_status == "Post-ovulation":
        return "Body signals suggest you may already be moving past the fertile window."
    if timing_status == "Fertility signal stronger than timing":
        return "Timing leans late-cycle, but current body signals still show stronger fertility signs."

    days_to_period = layer1.get("days_until_next_period")
    if isinstance(days_to_period, int) and days_to_period <= 3:
        return "Timing is close to the expected period window, but no period start was logged yet."

    return (
        f"History suggests {layer1_phase.lower()}, while body signals lean {layer2_phase.lower()}. "
        "The pattern is not specific enough to fully shift the timing interpretation."
    )


def get_layer3_output(layer1: Dict, layer2: Dict, period_start_logged: bool = False) -> Dict[str, object]:
    """Produce the Layer 3 timing annotation dict."""
    timing_status = get_timing_status(layer1, layer2, period_start_logged=period_start_logged)
    timing_note   = build_timing_note(layer1, layer2, timing_status, period_start_logged=period_start_logged)

    history_phase = layer1.get("top_phase", max(layer1["phase_probs"], key=layer1["phase_probs"].get))
    if history_phase == "Menstrual":
        history_phase = "Luteal"

    return {
        "timing_status":  timing_status,
        "timing_note":    timing_note,
        "history_phase":  history_phase,
        "symptom_phase":  layer2["top_phase"],
    }
