#!/usr/bin/env python3
"""
Cold-start scenario tests for the fusion pipeline.

These cover the highest-traffic real-world paths that have zero coverage
in the clinical validation suite (which only evaluates participants with
multiple logged period cycles).

Scenarios tested:
  1. Day 1 of app use — zero history, no symptoms
  2. Day 1 of app use — zero history, with symptoms
  3. Period start logged today (cycle reset)
  4. 1 period logged, currently cycle day 8, no symptoms
  5. 1 period logged, currently cycle day 8, with symptoms
  6. 2 periods logged, late luteal phase, high symptoms
  7. Ongoing menstrual (cycle day 3, flow substantial)
  8. Edge: very long cycle (45-day), cycle day 20
  9. Edge: very short cycle (21-day), cycle day 18
 10. Edge: irregular cycles (3 starts with high CV)

Usage:
    python tests/test_cold_start.py
    python tests/test_cold_start.py -v   # verbose — shows full output dicts
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cycle.fusion import get_fused_output

TODAY = "2024-06-15"   # fixed reference date for reproducibility


def _date(offset_days: int) -> str:
    """Return a date string relative to TODAY."""
    d = date.fromisoformat(TODAY) + timedelta(days=offset_days)
    return d.isoformat()


def _run(label: str, **kwargs) -> dict:
    """Run get_fused_output and return the result."""
    result = get_fused_output(**kwargs)
    return result


def check(label: str, result: dict, expected_phase: str | None = None,
          expected_mode: str | None = None, verbose: bool = False):
    """Assert expected values and print a pass/fail line."""
    failures = []

    if expected_phase and result.get("final_phase") != expected_phase:
        failures.append(
            f"phase: expected '{expected_phase}', got '{result.get('final_phase')}'"
        )
    if expected_mode and result.get("mode") != expected_mode:
        failures.append(
            f"mode: expected '{expected_mode}', got '{result.get('mode')}'"
        )

    # Sanity checks that apply to every scenario
    probs = result.get("final_phase_probs", {})
    prob_sum = sum(probs.values())
    if not (0.999 < prob_sum < 1.001):
        failures.append(f"probs don't sum to 1: {prob_sum:.4f}")
    if result.get("final_phase") not in ("Menstrual", "Follicular", "Fertility", "Luteal"):
        failures.append(f"unrecognised phase: {result.get('final_phase')}")

    status = "PASS" if not failures else "FAIL"
    print(f"  [{status}] {label}")
    for f in failures:
        print(f"         -> {f}")
    if verbose:
        print(f"         mode={result.get('mode')}  phase={result.get('final_phase')}  "
              f"probs={{{', '.join(f'{k}: {v:.2f}' for k, v in probs.items())}}}")

    return len(failures) == 0


def run_all(verbose: bool = False) -> int:
    """Run all cold-start scenarios. Returns number of failures."""
    failures = 0
    print("\n=== COLD-START SCENARIO TESTS ===\n")

    # ── Scenario 1: zero history, no symptoms ─────────────────────────────────
    r = _run("zero_history_no_symptoms",
             period_starts=[],
             symptoms=[],
             cervical_mucus="unknown",
             appetite=0, exerciselevel=0,
             today=TODAY)
    ok = check("Zero history, no symptoms — should not crash",
               r, verbose=verbose)
    failures += not ok

    # ── Scenario 2: zero history, with symptoms ───────────────────────────────
    r = _run("zero_history_with_symptoms",
             period_starts=[],
             symptoms=["cramps", "fatigue"],
             cervical_mucus="unknown",
             appetite=3, exerciselevel=2,
             today=TODAY)
    ok = check("Zero history, with symptoms — should not crash",
               r, verbose=verbose)
    failures += not ok

    # ── Scenario 3: period start logged today ─────────────────────────────────
    r = _run("period_start_today",
             period_starts=[TODAY],
             symptoms=["cramps"],
             cervical_mucus="unknown",
             period_start_logged=True,
             today=TODAY)
    ok = check("Period start logged today → Menstrual",
               r, expected_phase="Menstrual",
               expected_mode="period_start_override", verbose=verbose)
    failures += not ok

    # ── Scenario 4: 1 period logged, cycle day 8, no symptoms ────────────────
    period_1 = [_date(-7)]    # period started 7 days ago → currently day 8
    r = _run("one_period_day8_no_symptoms",
             period_starts=period_1,
             symptoms=[],
             cervical_mucus="unknown",
             appetite=0, exerciselevel=0,
             today=TODAY)
    ok = check("1 period logged, day 8, no symptoms → L1-only",
               r, expected_mode="layer1_only_non_menstrual", verbose=verbose)
    failures += not ok

    # ── Scenario 5: 1 period logged, cycle day 8, with symptoms ──────────────
    r = _run("one_period_day8_with_symptoms",
             period_starts=period_1,
             symptoms=["fatigue", "bloating"],
             cervical_mucus="watery",
             appetite=2, exerciselevel=1,
             today=TODAY)
    ok = check("1 period logged, day 8, with symptoms → fused or L1",
               r, verbose=verbose)
    failures += not ok

    # ── Scenario 6: 2 periods, late luteal, high symptom load ─────────────────
    two_periods = [_date(-56), _date(-28)]   # ~28-day cycles, now day 28
    r = _run("two_periods_late_luteal",
             period_starts=two_periods,
             symptoms=["cramps", "moodswing", "bloating", "fatigue"],
             cervical_mucus="dry",
             appetite=4, exerciselevel=1,
             today=TODAY,
             symptom_severities={
                 "cramps": 4, "moodswing": 4, "bloating": 3, "fatigue": 5,
                 "headaches": 0, "sorebreasts": 2, "sleepissue": 3,
                 "stress": 4, "foodcravings": 2, "indigestion": 1,
             })
    ok = check("2 periods, late luteal, high PMS symptoms → Luteal",
               r, expected_phase="Luteal", verbose=verbose)
    failures += not ok

    # ── Scenario 7: ongoing menstrual (day 3, substantial flow) ───────────────
    r = _run("ongoing_menstrual_day3",
             period_starts=[_date(-2)],
             symptoms=["cramps"],
             cervical_mucus="unknown",
             flow="moderate",
             today=TODAY)
    ok = check("Cycle day 3, moderate flow → Menstrual ongoing",
               r, expected_phase="Menstrual",
               expected_mode="menstrual_ongoing", verbose=verbose)
    failures += not ok

    # ── Scenario 8: edge — very long cycle (45-day), day 20 ──────────────────
    long_cycle_starts = [_date(-90), _date(-45)]   # 45-day cycles, now day 20
    r = _run("long_cycle_day20",
             period_starts=long_cycle_starts,
             symptoms=["fatigue"],
             cervical_mucus="unknown",
             appetite=2, exerciselevel=2,
             today=TODAY)
    ok = check("Long cycle (45d), day 20 — should not crash",
               r, verbose=verbose)
    failures += not ok

    # ── Scenario 9: edge — short cycle (21-day), day 18 ──────────────────────
    short_cycle_starts = [_date(-42), _date(-21)]  # 21-day cycles, now day 18
    r = _run("short_cycle_day18",
             period_starts=short_cycle_starts,
             symptoms=["moodswing", "sorebreasts"],
             cervical_mucus="dry",
             appetite=3, exerciselevel=2,
             today=TODAY)
    ok = check("Short cycle (21d), day 18 — should predict Luteal",
               r, expected_phase="Luteal", verbose=verbose)
    failures += not ok

    # ── Scenario 10: irregular cycles (high CV) ───────────────────────────────
    irregular_starts = [_date(-100), _date(-65), _date(-28)]  # 35d, 37d cycles
    r = _run("irregular_cycles",
             period_starts=irregular_starts,
             symptoms=["headaches", "fatigue", "stress"],
             cervical_mucus="unknown",
             appetite=3, exerciselevel=2,
             today=TODAY)
    ok = check("Irregular cycles — should not crash, returns a valid phase",
               r, verbose=verbose)
    failures += not ok

    # ── Scenario 11: all symptoms logged at maximum severity ─────────────────
    r = _run("max_symptoms",
             period_starts=[_date(-14)],
             symptoms=["cramps","fatigue","moodswing","bloating","headaches",
                       "sorebreasts","sleepissue","stress","foodcravings","indigestion"],
             cervical_mucus="sticky",
             appetite=5, exerciselevel=5,
             today=TODAY,
             symptom_severities={s: 5 for s in [
                 "cramps","fatigue","moodswing","bloating","headaches",
                 "sorebreasts","sleepissue","stress","foodcravings","indigestion"]})
    ok = check("All symptoms at max severity — should not crash",
               r, verbose=verbose)
    failures += not ok

    # ── Scenario 12: single-day history (period started today, next day inference)
    r = _run("minimal_history_day2",
             period_starts=[_date(-1)],   # started yesterday
             symptoms=[],
             cervical_mucus="unknown",
             flow="moderate",
             today=TODAY)
    ok = check("Period start yesterday (day 2), moderate flow → Menstrual",
               r, expected_phase="Menstrual", verbose=verbose)
    failures += not ok

    print(f"\n{'='*40}")
    passed = 12 - failures
    print(f"  Results: {passed}/12 passed, {failures}/12 failed")
    print(f"{'='*40}\n")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show full phase probabilities and mode for each scenario")
    args = parser.parse_args()
    exit_code = run_all(verbose=args.verbose)
    sys.exit(exit_code)
