"""
Ordinal rank tables for all recommender input signals.

These dicts map human-readable severity labels to integer ranks so that
rule conditions like ``ENERGY >= Low`` can be evaluated numerically.
Each signal has its own scale; the ``SIGNAL_RANKS`` registry maps signal
names to their respective rank dict.
"""

ENERGY_RANK: dict[str, int] = {
    "Very Low": 0, "Low": 1, "Medium": 2, "High": 3, "Very High": 4,
}
STRESS_RANK: dict[str, int] = {
    "Low": 0, "Moderate": 1, "High": 2, "Severe": 3,
}
MOOD_RANK: dict[str, int] = {
    "Very Low": 0, "Low": 1, "Medium": 2, "High": 3, "Very High": 4,
}
CRAMP_RANK: dict[str, int] = {
    "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Very Severe": 4,
}
BLOATING_RANK: dict[str, int] = {
    "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3,
}
SLEEP_RANK: dict[str, int] = {
    "Poor": 0, "Fair": 1, "Good": 2, "Very Good": 3,
}
IRRITABILITY_RANK: dict[str, int] = {
    "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3,
}
MOOD_SWING_RANK: dict[str, int] = {
    "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3,
}
CRAVINGS_RANK: dict[str, int] = {
    "None": 0, "Mild": 1, "Moderate": 2, "Strong": 3,
}
FLOW_RANK: dict[str, int] = {
    "None": 0, "Light": 1, "Moderate": 2, "Heavy": 3, "Very Heavy": 4,
}
TENDERNESS_RANK: dict[str, int] = {
    "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3,
}
INDIGESTION_RANK: dict[str, int] = {
    "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3,
}
BRAIN_FOG_RANK: dict[str, int] = {
    "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3,
}

# Registry: signal name → rank dict
SIGNAL_RANKS: dict[str, dict[str, int]] = {
    "ENERGY":            ENERGY_RANK,
    "STRESS":            STRESS_RANK,
    "MOOD":              MOOD_RANK,
    "CRAMP":             CRAMP_RANK,
    "BLOATING":          BLOATING_RANK,
    "SLEEP_QUALITY":     SLEEP_RANK,
    "IRRITABILITY":      IRRITABILITY_RANK,
    "MOOD_SWING":        MOOD_SWING_RANK,
    "CRAVINGS":          CRAVINGS_RANK,
    "FLOW":              FLOW_RANK,
    "BREAST_TENDERNESS": TENDERNESS_RANK,
    "INDIGESTION":       INDIGESTION_RANK,
    "BRAIN_FOG":         BRAIN_FOG_RANK,
}
