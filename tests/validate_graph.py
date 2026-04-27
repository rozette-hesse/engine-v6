"""
Graph Recommender Validation
=============================
Run: python tests/validate_graph.py

Runs all 25 clinical scenarios against the graph recommender.
This is the primary CI check for recommendation engine changes.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import recommend
from tests.validate import SCENARIOS, check, TOP_N, FETCH_N


def run():
    passed = 0
    failed = 0
    total  = len(SCENARIOS)

    print(f"\nGraph Recommender — Validation ({total} scenarios)\n{'─'*65}")

    for scenario in SCENARIOS:
        failures = check(scenario, recommend(scenario["inputs"], top_n=FETCH_N))

        if failures:
            failed += 1
            print(f"❌  {scenario['name']}")
            for f in failures:
                print(f)
        else:
            passed += 1
            print(f"✅  {scenario['name']}")

    print(f"\n{'─'*65}")
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED")
    else:
        print("  ✓  all clear")
    print()

    return failed == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
