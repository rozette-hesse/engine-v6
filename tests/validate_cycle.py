"""
Cycle Engine — Golden Scenario Regression Tests
================================================
Run:  python tests/validate_cycle.py
      python tests/validate_cycle.py -v    # verbose: print full output dicts

Covers all four fusion modes, the menstrual-continuation detector, the
3-day rolling-history path, and phase accuracy across the full cycle.
This is the canonical regression suite for any change to:
  cycle/fusion.py, cycle/layer1_period.py, cycle/layer2/*, cycle/config.py

Exit codes
----------
  0 — all scenarios passed
  1 — one or more scenarios failed

Design notes
------------
  * TODAY = "2024-06-15" is fixed so results are deterministic.
  * Scenarios in the ROLLING_HISTORY group pass ``recent_daily_logs`` to
    exercise the 3-day history path — these are the P0 accuracy fix.
  * Pure-logic scenarios (period override, menstrual detector) have no
    model dependency; they must always pass.
  * Model-dependent scenarios (Fertility, fused Luteal) were verified
    against the installed v4 artifacts and should be treated as regression
    baselines: a failure here means something changed in fusion or artifacts.
"""

import sys
import os
import warnings
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress sklearn feature-name warnings from LightGBM inference
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

from cycle.fusion import get_fused_output
from cycle.config import PHASES

TODAY = "2024-06-15"
VERBOSE = "-v" in sys.argv


# ── Helpers ───────────────────────────────────────────────────────────────────

def _period_starts(cycle_day: int, cycle_length: int = 28, n_prior: int = 2) -> List[str]:
    """
    Return n_prior+1 period start dates so that ``TODAY`` falls on
    cycle day ``cycle_day`` of the most recent cycle.

    Example: cycle_day=14, cycle_length=28 → latest start is June 2nd,
    with two prior starts 28 days apart.
    """
    latest = date.fromisoformat(TODAY) - timedelta(days=cycle_day - 1)
    starts = [latest]
    for _ in range(n_prior):
        starts.append(starts[-1] - timedelta(days=cycle_length))
    return sorted(s.isoformat() for s in starts)


def _run(name: str, checks: List[Tuple[str, Any, Any]], **kwargs) -> bool:
    """
    Run get_fused_output(**kwargs), apply assertion tuples, return True if all pass.

    Each check is (path, expected, actual_func) where path is a dot-path into
    the output dict or a special key like "final_phase", "mode".
    """
    out = get_fused_output(today=TODAY, **kwargs)

    if VERBOSE:
        import json
        print(f"\n  [verbose] {name}")
        print(f"  {json.dumps(out, indent=4, default=str)}")

    failures = []
    for label, expected, actual in checks:
        if isinstance(expected, set):
            if actual not in expected:
                failures.append(f"  FAIL {label}: expected one of {expected}, got {actual!r}")
        elif callable(expected):
            if not expected(actual):
                failures.append(f"  FAIL {label}: predicate failed for value {actual!r}")
        else:
            if actual != expected:
                failures.append(f"  FAIL {label}: expected {expected!r}, got {actual!r}")

    if failures:
        print(f"\u274c  {name}")
        for f in failures:
            print(f)
        return False

    print(f"\u2705  {name}")
    return True


def _get(out_fn, *keys):
    """
    Lazy extractor: returns a lambda that calls out_fn(), then walks key path.
    Use with _run() by passing a 0-arg wrapper that returns the result dict.
    """
    # Not used directly; we inline field access instead for clarity.
    pass


# ── Scenario runner ────────────────────────────────────────────────────────────

def run_all() -> bool:
    passed = 0
    failed = 0

    def s(name: str, checks_fn, **kwargs) -> None:
        """Run one scenario and record result."""
        nonlocal passed, failed
        out = get_fused_output(today=TODAY, **kwargs)

        if VERBOSE:
            import json
            print(f"\n  [verbose] {name}")
            print(f"  {json.dumps(out, indent=4, default=str)}")

        failures = checks_fn(out)
        if failures:
            failed += 1
            print(f"\u274c  {name}")
            for f in failures:
                print(f)
        else:
            passed += 1
            print(f"\u2705  {name}")

    print(f"\nCycle Engine — Golden Scenario Validation\n{'─'*65}")

    # ── Group 1: period_start_override mode ───────────────────────────────────
    print("\n  [Mode: period_start_override]")

    def _check_period_override(out):
        errs = []
        if out["mode"] != "period_start_override":
            errs.append(f"  FAIL mode: expected 'period_start_override', got {out['mode']!r}")
        if out["final_phase"] != "Menstrual":
            errs.append(f"  FAIL final_phase: expected 'Menstrual', got {out['final_phase']!r}")
        if out["final_phase_probs"]["Menstrual"] != 1.0:
            errs.append("  FAIL final_phase_probs.Menstrual: expected 1.0")
        if out["layer2"] is not None:
            errs.append("  FAIL layer2: expected None (skipped)")
        return errs

    s("MEN_PERIOD_START_OVERRIDE",
      _check_period_override,
      period_starts=_period_starts(25),
      period_start_logged=True)

    s("MEN_PERIOD_START_OVERRIDE_FIRST_EVER",
      _check_period_override,
      period_starts=["2024-06-15"],
      period_start_logged=True)

    # ── Group 2: menstrual_ongoing mode ───────────────────────────────────────
    print("\n  [Mode: menstrual_ongoing]")

    def _check_men_ongoing(out):
        errs = []
        if out["mode"] != "menstrual_ongoing":
            errs.append(f"  FAIL mode: expected 'menstrual_ongoing', got {out['mode']!r}")
        if out["final_phase"] != "Menstrual":
            errs.append(f"  FAIL final_phase: expected 'Menstrual', got {out['final_phase']!r}")
        if out["layer2"] is not None:
            errs.append("  FAIL layer2: expected None (skipped)")
        return errs

    s("MEN_ONGOING_DAY2",
      _check_men_ongoing,
      period_starts=_period_starts(2),
      flow="heavy")

    s("MEN_ONGOING_DAY4_HEAVY",
      _check_men_ongoing,
      period_starts=_period_starts(4),
      flow="heavy")

    s("MEN_ONGOING_DAY4_NO_FLOW",
      _check_men_ongoing,
      period_starts=_period_starts(4),
      flow="none")

    s("MEN_ONGOING_DAY5_MODERATE",
      _check_men_ongoing,
      period_starts=_period_starts(5),
      flow="moderate")

    s("MEN_ONGOING_DAY5_NO_FLOW",
      _check_men_ongoing,
      period_starts=_period_starts(5),
      flow="none")

    s("MEN_ONGOING_DAY6_SUBSTANTIAL",
      _check_men_ongoing,
      period_starts=_period_starts(6),
      flow="heavy")

    # ── Group 3: menstrual detector rejects edge cases ────────────────────────
    print("\n  [Menstrual detector — boundary cases]")

    def _check_not_menstrual_ongoing(out):
        if out["mode"] == "menstrual_ongoing":
            return [f"  FAIL mode: expected anything except 'menstrual_ongoing', got that"]
        return []

    # Day 5: spotting exits the menstrual window (none/substantial/missing do not)
    s("MEN_STOPS_AT_DAY5_SPOTTING",
      _check_not_menstrual_ongoing,
      period_starts=_period_starts(5),
      flow="spotting")

    s("MEN_STOPS_AT_DAY8_ANY_FLOW",
      _check_not_menstrual_ongoing,
      period_starts=_period_starts(8),
      flow="heavy")

    # ── Group 4: rolling history — day 7 menstrual continuation ──────────────
    # This is the P0 fix: the rule at cycle_day 7 requires yesterday's flow
    # to be substantial.  Without recent_daily_logs the rule cannot fire.
    print("\n  [Rolling history — 3-day context]")

    s("MEN_ONGOING_DAY7_WITH_HISTORY",
      _check_men_ongoing,
      period_starts=_period_starts(7),
      flow="heavy",
      recent_daily_logs=[{"flow": "heavy", "symptoms": [], "cervical_mucus": "unknown"}])

    s("MEN_STOPPED_DAY7_NO_HISTORY",
      _check_not_menstrual_ongoing,
      period_starts=_period_starts(7),
      flow="heavy",
      recent_daily_logs=[])

    def _check_fer_fertility(out):
        errs = []
        if out["final_phase"] != "Fertility":
            errs.append(f"  FAIL final_phase: expected 'Fertility', got {out['final_phase']!r}")
        if out["mode"] != "fused_non_menstrual":
            errs.append(f"  FAIL mode: expected 'fused_non_menstrual', got {out['mode']!r}")
        return errs

    s("FER_ROLLING_HISTORY_CREAMY_THEN_EGGWHITE",
      _check_fer_fertility,
      period_starts=_period_starts(13),
      cervical_mucus="eggwhite",
      recent_daily_logs=[
          {"cervical_mucus": "creamy", "symptoms": [], "flow": "none"},
      ])

    # ── Group 5: layer1_only_no_history ───────────────────────────────────────
    print("\n  [Mode: layer1_only_no_history]")

    def _check_no_history(out):
        errs = []
        if out["mode"] != "layer1_only_no_history":
            errs.append(f"  FAIL mode: expected 'layer1_only_no_history', got {out['mode']!r}")
        if out["final_phase"] not in {"Follicular", "Fertility", "Luteal"}:
            errs.append(f"  FAIL final_phase: {out['final_phase']!r} not in non-menstrual phases")
        if out["final_phase_probs"].get("Menstrual", 0) != 0.0:
            errs.append("  FAIL final_phase_probs.Menstrual: expected 0.0 (no menstrual without history)")
        return errs

    s("NO_HISTORY_NO_SYMPTOMS",
      _check_no_history,
      period_starts=[])

    s("NO_HISTORY_WITH_SYMPTOMS",
      _check_no_history,
      period_starts=[],
      symptoms=["fatigue", "cramps"])

    # ── Group 6: layer1_only_non_menstrual ────────────────────────────────────
    print("\n  [Mode: layer1_only_non_menstrual — phase accuracy]")

    def _check_l1_only(expected_phase):
        def _check(out):
            errs = []
            if out["mode"] != "layer1_only_non_menstrual":
                errs.append(f"  FAIL mode: expected 'layer1_only_non_menstrual', got {out['mode']!r}")
            if out["final_phase"] != expected_phase:
                errs.append(f"  FAIL final_phase: expected {expected_phase!r}, got {out['final_phase']!r}")
            return errs
        return _check

    s("FOL_L1_ONLY_DAY8",
      _check_l1_only("Follicular"),
      period_starts=_period_starts(8))

    s("LUT_L1_ONLY_DAY22",
      _check_l1_only("Luteal"),
      period_starts=_period_starts(22))

    # ── Group 7: fused_non_menstrual — model-dependent ────────────────────────
    print("\n  [Mode: fused_non_menstrual — model accuracy]")

    s("FER_EGGWHITE_MUCUS_DAY14",
      _check_fer_fertility,
      period_starts=_period_starts(14),
      cervical_mucus="eggwhite")

    def _check_lut_fused(out):
        errs = []
        if out["mode"] != "fused_non_menstrual":
            errs.append(f"  FAIL mode: expected 'fused_non_menstrual', got {out['mode']!r}")
        if out["final_phase"] != "Luteal":
            errs.append(f"  FAIL final_phase: expected 'Luteal', got {out['final_phase']!r}")
        return errs

    s("LUT_CLASSIC_SYMPTOMS_DAY21",
      _check_lut_fused,
      period_starts=_period_starts(21),
      symptoms=["sorebreasts", "foodcravings", "bloating"])

    s("LUT_STRONG_SYMPTOMS_DAY25",
      _check_lut_fused,
      period_starts=_period_starts(25),
      symptoms=["cramps", "moodswing", "fatigue", "bloating", "foodcravings"])

    def _check_fol_fused(out):
        errs = []
        if out["mode"] != "fused_non_menstrual":
            errs.append(f"  FAIL mode: expected 'fused_non_menstrual', got {out['mode']!r}")
        if out["final_phase"] != "Follicular":
            errs.append(f"  FAIL final_phase: expected 'Follicular', got {out['final_phase']!r}")
        return errs

    s("FOL_LOW_BURDEN_DAY9",
      _check_fol_fused,
      period_starts=_period_starts(9),
      symptoms=["fatigue"],
      cervical_mucus="dry")

    # ── Group 8: irregular cycle — low forecast confidence ────────────────────
    # Cycle lengths [21, 35, 25] → std ≈ 5.9 > 5 → forecast_confidence = "low"
    # The constrain layer opens all phases when confidence is low.
    print("\n  [Irregular cycle — low forecast confidence]")

    IRREGULAR_STARTS = ["2024-03-17", "2024-04-07", "2024-05-12", "2024-06-06"]

    def _check_irregular(out):
        errs = []
        conf = out.get("layer1", {}).get("forecast_confidence")
        if conf != "low":
            errs.append(f"  FAIL forecast_confidence: expected 'low', got {conf!r}")
        if out["mode"] != "fused_non_menstrual":
            errs.append(f"  FAIL mode: expected 'fused_non_menstrual', got {out['mode']!r}")
        return errs

    s("IRREGULAR_CYCLE_LOW_CONFIDENCE",
      _check_irregular,
      period_starts=IRREGULAR_STARTS,
      symptoms=["fatigue"])

    # ── Group 9: output contract ──────────────────────────────────────────────
    print("\n  [Output contract]")

    def _check_output_shape(out):
        errs = []
        required_keys = {"mode", "layer1", "layer2", "layer3", "final_phase_probs", "final_phase"}
        missing = required_keys - set(out.keys())
        if missing:
            errs.append(f"  FAIL missing output keys: {missing}")

        prob_sum = sum(out.get("final_phase_probs", {}).values())
        if abs(prob_sum - 1.0) > 1e-6:
            errs.append(f"  FAIL final_phase_probs sum: expected 1.0, got {prob_sum}")

        phase = out.get("final_phase")
        if phase not in PHASES:
            errs.append(f"  FAIL final_phase not in PHASES: got {phase!r}")

        l1 = out.get("layer1", {})
        if not l1.get("cycle_day"):
            errs.append("  FAIL layer1.cycle_day missing or zero")
        return errs

    s("OUTPUT_SHAPE_FUSED",
      _check_output_shape,
      period_starts=_period_starts(14),
      symptoms=["fatigue"],
      cervical_mucus="eggwhite")

    s("OUTPUT_SHAPE_L1_ONLY",
      _check_output_shape,
      period_starts=_period_starts(10))

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'─'*65}")
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED")
    else:
        print("  \u2713  all clear")
    print()

    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
