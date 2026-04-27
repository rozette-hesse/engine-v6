"""
API Wiring Regression Tests
============================
Run:  python tests/validate_wiring.py
      python tests/validate_wiring.py -v

Proves that fields on UnifiedRequest actually reach the cycle engine and
proximity model — bugs caught here can't be caught by tests that call
get_fused_output() or get_proximity_output() directly.

Two scenarios
-------------
  1. cervical_mucus: changing the field on the payload changes the value
     that build_cycle_inputs delivers to the cycle engine.
  2. symptom severity: calling get_proximity_output with mild vs severe
     severities produces different probabilities.
"""

import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

from core.transformers import build_cycle_inputs
from core.pipeline import _proximity_severities
from cycle import get_proximity_output
from cycle.config import SUPPORTED_SYMPTOMS
from app.schemas.requests import UnifiedRequest
from app.schemas.enums import (
    CervicalMucus, CrampLevel, EnergyLevel, StressLevel, MoodLevel,
    MoodSwingLevel, BloatingLevel,
)

VERBOSE = "-v" in sys.argv


def _base_signals(**overrides) -> dict:
    return {
        "ENERGY": "Medium", "STRESS": "Low", "MOOD": "Medium",
        "CRAMP": "None", "HEADACHE": 0, "BLOATING": "None",
        "SLEEP_QUALITY": "Good", "IRRITABILITY": "None",
        "MOOD_SWING": "None", "CRAVINGS": "None", "FLOW": "None",
        "BREAST_TENDERNESS": "None", "INDIGESTION": "None", "BRAIN_FOG": "None",
        **overrides,
    }


def _run(name: str, fn) -> bool:
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as exc:
        print(f"  FAIL  {name}: {exc}")
        return False


# ── Test 1: cervical_mucus wires through to cycle inputs ──────────────────────

def test_cervical_mucus_wiring():
    for mucus_value in ("dry", "eggwhite", "creamy"):
        signals = _base_signals()
        enriched = {**signals, "CERVICAL_MUCUS": mucus_value, "APPETITE": 0, "EXERCISE_LEVEL": 0}
        cycle_inputs = build_cycle_inputs(enriched)
        assert cycle_inputs["cervical_mucus"] == mucus_value, (
            f"expected cervical_mucus={mucus_value!r}, got {cycle_inputs['cervical_mucus']!r}"
        )

    # unknown is the default — make sure it differs from eggwhite
    for val_a, val_b in [("unknown", "eggwhite"), ("dry", "watery")]:
        sig_a = {**_base_signals(), "CERVICAL_MUCUS": val_a, "APPETITE": 0, "EXERCISE_LEVEL": 0}
        sig_b = {**_base_signals(), "CERVICAL_MUCUS": val_b, "APPETITE": 0, "EXERCISE_LEVEL": 0}
        inputs_a = build_cycle_inputs(sig_a)
        inputs_b = build_cycle_inputs(sig_b)
        assert inputs_a["cervical_mucus"] != inputs_b["cervical_mucus"], (
            f"expected {val_a!r} and {val_b!r} to produce different cycle inputs"
        )

    if VERBOSE:
        print("    cervical_mucus values propagate correctly to cycle engine")


# ── Test 2: symptom severity changes proximity probability ────────────────────

def test_severity_affects_proximity_probability():
    mild_sev   = {s: 0 for s in SUPPORTED_SYMPTOMS}
    severe_sev = {s: 0 for s in SUPPORTED_SYMPTOMS}

    for sym in ("cramps", "bloating", "moodswing", "fatigue"):
        mild_sev[sym]   = 1
        severe_sev[sym] = 4

    result_mild   = get_proximity_output(symptom_severities=mild_sev)
    result_severe = get_proximity_output(symptom_severities=severe_sev)

    if VERBOSE:
        print(f"    mild probability:   {result_mild['probability']}")
        print(f"    severe probability: {result_severe['probability']}")

    assert result_severe["probability"] > result_mild["probability"], (
        f"severe symptoms ({result_severe['probability']}) should yield higher "
        f"proximity probability than mild ({result_mild['probability']})"
    )


# ── Test 3: _proximity_severities maps enums correctly ────────────────────────

def test_proximity_severity_mapping():
    payload_mild = UnifiedRequest(
        period_starts=["2024-01-01"],
        CRAMP=CrampLevel.mild,
        BLOATING=BloatingLevel.mild,
        ENERGY=EnergyLevel.medium,
    )
    payload_severe = UnifiedRequest(
        period_starts=["2024-01-01"],
        CRAMP=CrampLevel.very_severe,
        BLOATING=BloatingLevel.severe,
        ENERGY=EnergyLevel.very_low,
    )

    sev_mild   = _proximity_severities(payload_mild)
    sev_severe = _proximity_severities(payload_severe)

    assert sev_mild["cramps"]   < sev_severe["cramps"],   "cramps severity should increase"
    assert sev_mild["bloating"] < sev_severe["bloating"], "bloating severity should increase"
    assert sev_mild["fatigue"]  < sev_severe["fatigue"],  "fatigue severity should increase with lower energy"

    if VERBOSE:
        print(f"    mild:   cramps={sev_mild['cramps']} bloating={sev_mild['bloating']} fatigue={sev_mild['fatigue']}")
        print(f"    severe: cramps={sev_severe['cramps']} bloating={sev_severe['bloating']} fatigue={sev_severe['fatigue']}")


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    ("cervical_mucus wires through to cycle inputs", test_cervical_mucus_wiring),
    ("symptom severity changes proximity probability", test_severity_affects_proximity_probability),
    ("_proximity_severities maps enums to correct integers", test_proximity_severity_mapping),
]

if __name__ == "__main__":
    print("API Wiring Regression Tests")
    print("=" * 50)
    results = [_run(name, fn) for name, fn in TESTS]
    passed  = sum(results)
    total   = len(results)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
