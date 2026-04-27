#!/usr/bin/env python3
"""
Cycle Engine Validation — scripts entry point.

Runs the golden-scenario regression suite from tests/validate_cycle.py.

Usage
-----
  python scripts/validate_cycle.py          # run golden scenarios
  python scripts/validate_cycle.py -v       # verbose output

Dataset-level evaluation
------------------------
To evaluate against a real participant dataset, extend this script by
loading your labelled data and calling get_fused_output() day-by-day,
passing the previous 1–3 days' logs as ``recent_daily_logs`` so the
engine has rolling context (the P0 accuracy fix).

Minimal example::

    from cycle.fusion import get_fused_output

    for participant in dataset:
        recent = []
        for day in participant.days_sorted():
            out = get_fused_output(
                period_starts=day.period_starts,
                symptoms=day.symptoms,
                cervical_mucus=day.cervical_mucus,
                flow=day.flow,
                today=day.date,
                recent_daily_logs=recent[-3:],   # <-- rolling history
            )
            recent.append({
                "symptoms":       day.symptoms,
                "cervical_mucus": day.cervical_mucus,
                "flow":           day.flow,
            })
            yield day.date, day.true_phase, out["final_phase"]
"""

import sys
import os

# Ensure project root is on the path when run from scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.validate_cycle import run_all

if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
