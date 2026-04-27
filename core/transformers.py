"""
Signal Transformers
===================
Converts the raw user input format (severity-rated enum strings) into the
internal formats that downstream engines expect:

  1. Binary symptom list + extras  → cycle engine (Layer 2 legacy input)
  2. Ordinal severity dict         → cycle engine (Layer 2 preferred input)
  3. Severity-keyed signal dict    → recommendation engine

The UI collects severity ratings. This module translates those into the
binary presence/absence flags the ML model was trained on, the 0-5 ordinal
severity dict the v3/v4 feature builder prefers, and the flat dict the
rule-based recommender consumes.
"""

from typing import Dict, List


# ── Severity → 0-5 ordinal mappings ──────────────────────────────────────────
# Used by build_symptom_severities() and re-exported for pipeline use.

_SYMPTOM_SEV: Dict[str, int] = {
    "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Very Severe": 4, "Strong": 3,
}
_FATIGUE_FROM_ENERGY: Dict[str, int] = {
    "Very Low": 4, "Low": 3, "Medium": 1, "High": 0, "Very High": 0,
}
_SLEEPISSUE_FROM_SLEEP: Dict[str, int] = {
    "Poor": 3, "Fair": 2, "Good": 0, "Very Good": 0,
}
_STRESS_SEV: Dict[str, int] = {
    "Low": 0, "Moderate": 2, "High": 3, "Severe": 4,
}


# ── Severity → presence mapping ───────────────────────────────────────────────
# Any value at or above the threshold counts as "present" for Layer 2.

_SEVERITY_PRESENT: Dict[str, bool] = {
    "None":        False,
    "Mild":        True,
    "Moderate":    True,
    "Severe":      True,
    "Very Severe": True,
    "Strong":      True,  # cravings scale
    "Light":       True,
    "Heavy":       True,
    "Very Heavy":  True,
}


def _is_present(value: str) -> bool:
    """Return True if the severity string indicates the symptom is active."""
    return _SEVERITY_PRESENT.get(value, False)


def severity_to_binary_symptoms(signals: Dict[str, object]) -> List[str]:
    """
    Convert severity-rated signals to the binary symptom names Layer 2 expects.

    Mapping
    -------
    HEADACHE (0–10 int)        → "headaches"   if > 0
    CRAMP severity             → "cramps"
    BREAST_TENDERNESS          → "sorebreasts"
    ENERGY (Low / Very Low)    → "fatigue"
    SLEEP_QUALITY (Poor/Fair)  → "sleepissue"
    MOOD_SWING severity        → "moodswing"
    STRESS (High / Severe)     → "stress"
    CRAVINGS severity          → "foodcravings"
    INDIGESTION severity       → "indigestion"
    BLOATING severity          → "bloating"
    """
    symptoms: List[str] = []

    if int(signals.get("HEADACHE", 0)) > 0:
        symptoms.append("headaches")

    _SEVERITY_MAP = {
        "CRAMP":             "cramps",
        "BREAST_TENDERNESS": "sorebreasts",
        "MOOD_SWING":        "moodswing",
        "CRAVINGS":          "foodcravings",
        "INDIGESTION":       "indigestion",
        "BLOATING":          "bloating",
    }
    for signal_key, symptom_name in _SEVERITY_MAP.items():
        if _is_present(str(signals.get(signal_key, "None"))):
            symptoms.append(symptom_name)

    if str(signals.get("ENERGY", "Medium")) in ("Low", "Very Low"):
        symptoms.append("fatigue")

    if str(signals.get("SLEEP_QUALITY", "Good")) in ("Poor", "Fair"):
        symptoms.append("sleepissue")

    if str(signals.get("STRESS", "Low")) in ("High", "Severe"):
        symptoms.append("stress")

    return symptoms


def build_cycle_inputs(signals: Dict[str, object]) -> Dict[str, object]:
    """
    Build keyword arguments for ``cycle.get_fused_output()`` from raw signals.

    Note: CERVICAL_MUCUS, APPETITE, and EXERCISE_LEVEL are passed directly
    from the unified request to signals before calling this function.
    """
    return {
        "symptoms":        severity_to_binary_symptoms(signals),
        "cervical_mucus":  str(signals.get("CERVICAL_MUCUS", "unknown")),
        "appetite":        int(signals.get("APPETITE", 0)),
        "exerciselevel":   int(signals.get("EXERCISE_LEVEL", 0)),
        "flow":            str(signals.get("FLOW", "none")),
    }


def build_symptom_severities(payload) -> Dict[str, int]:
    """
    Convert a request payload to a {symptom: 0-5} ordinal severity dict.

    This is the canonical symptom representation for the v3/v4 feature builder
    and for persistent storage.  The same mapping is used by the proximity model.

    Parameters
    ----------
    payload : UnifiedRequest (or any object with the same symptom attributes)
    """
    return {
        "headaches":    min(5, round(int(getattr(payload, "HEADACHE", 0)) / 2)),
        "cramps":       _SYMPTOM_SEV.get(payload.CRAMP.value,             0),
        "sorebreasts":  _SYMPTOM_SEV.get(payload.BREAST_TENDERNESS.value, 0),
        "fatigue":      _FATIGUE_FROM_ENERGY.get(payload.ENERGY.value,    0),
        "sleepissue":   _SLEEPISSUE_FROM_SLEEP.get(payload.SLEEP_QUALITY.value, 0),
        "moodswing":    _SYMPTOM_SEV.get(payload.MOOD_SWING.value,        0),
        "stress":       _STRESS_SEV.get(payload.STRESS.value,             0),
        "foodcravings": _SYMPTOM_SEV.get(payload.CRAVINGS.value,          0),
        "indigestion":  _SYMPTOM_SEV.get(payload.INDIGESTION.value,       0),
        "bloating":     _SYMPTOM_SEV.get(payload.BLOATING.value,          0),
    }


def build_recommender_inputs(signals: Dict[str, object], phase_code: str) -> Dict[str, object]:
    """
    Build keyword arguments for ``recommender.recommend()`` from raw signals.

    Phase code comes from the phase resolver (MEN / FOL / OVU / LUT / LL).
    """
    return {
        "PHASE":             phase_code,
        "ENERGY":            str(signals.get("ENERGY",            "Medium")),
        "STRESS":            str(signals.get("STRESS",            "Low")),
        "MOOD":              str(signals.get("MOOD",              "Medium")),
        "CRAMP":             str(signals.get("CRAMP",             "None")),
        "HEADACHE":          int(signals.get("HEADACHE",          0)),
        "BLOATING":          str(signals.get("BLOATING",          "None")),
        "SLEEP_QUALITY":     str(signals.get("SLEEP_QUALITY",     "Good")),
        "IRRITABILITY":      str(signals.get("IRRITABILITY",      "None")),
        "MOOD_SWING":        str(signals.get("MOOD_SWING",        "None")),
        "CRAVINGS":          str(signals.get("CRAVINGS",          "None")),
        "FLOW":              str(signals.get("FLOW",              "None")),
        "BREAST_TENDERNESS": str(signals.get("BREAST_TENDERNESS", "None")),
        "INDIGESTION":       str(signals.get("INDIGESTION",       "None")),
        "BRAIN_FOG":         str(signals.get("BRAIN_FOG",         "None")),
    }
