"""
InBalance Clinical Validation Matrix
=====================================
Run: python tests/validate.py

Defines 25 golden clinical scenarios and asserts expected outcomes.
A scenario fails if any of its assertions are violated.

Assertion types
---------------
  top_tag     — at least one unblocked top-N item must carry this tag
  not_top_tag — no unblocked top-N item may carry this tag
  blocked_tag — at least one item with this tag must be blocked
  score_gap   — top score of group A must exceed top score of group B by min_gap
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import recommend
from collections import defaultdict

TOP_N   = 3   # "visible top-N" used by top_tag / not_top_tag assertions
FETCH_N = 50  # large enough to retrieve every item, including blocked ones


# ── Assertion helpers ─────────────────────────────────────────────────────────

def get_group(results, domain, category):
    return results.get(domain, {}).get(category, [])


def top_unblocked(items, n=TOP_N):
    """Return the first n unblocked items (engine already sorted unblocked-first by score)."""
    return [i for i in items if not i["is_blocked"]][:n]


def tags_of(items):
    return {tag for item in items for tag in item.get("tags", [])}


def top_score(items):
    visible = [i for i in items if not i["is_blocked"]]
    return max((i["final_score"] for i in visible), default=0)


def check(scenario, results):
    failures = []
    for a in scenario["assertions"]:
        t     = a["type"]
        items = get_group(results, a["domain"], a.get("category", ""))

        if t == "top_tag":
            if a["tag"] not in tags_of(top_unblocked(items)):
                failures.append(
                    f"  FAIL top_tag: '{a['tag']}' not in unblocked top-{TOP_N} of "
                    f"{a['domain']}/{a['category']}"
                )

        elif t == "not_top_tag":
            found = [i["text"][:50] for i in top_unblocked(items) if a["tag"] in i.get("tags", [])]
            if found:
                failures.append(
                    f"  FAIL not_top_tag: '{a['tag']}' found in unblocked top-{TOP_N} of "
                    f"{a['domain']}/{a['category']}: {found}"
                )

        elif t == "blocked_tag":
            blocked = [i for i in items if i["is_blocked"] and a["tag"] in i.get("tags", [])]
            if not blocked:
                failures.append(
                    f"  FAIL blocked_tag: no item with tag '{a['tag']}' is blocked in "
                    f"{a['domain']}/{a['category']}"
                )

        elif t == "score_gap":
            items_a = get_group(results, a["domain"], a["cat_high"])
            items_b = get_group(results, a["domain"], a["cat_low"])
            gap     = top_score(items_a) - top_score(items_b)
            if gap < a["min_gap"]:
                failures.append(
                    f"  FAIL score_gap: {a['domain']}/{a['cat_high']} should beat "
                    f"{a['cat_low']} by ≥{a['min_gap']}, got gap={gap}"
                )

    return failures


# ── Golden scenarios ──────────────────────────────────────────────────────────

SCENARIOS = [

    # 1. MEN baseline: rest, iron, breathwork
    {
        "name": "MEN_BASELINE",
        "inputs": {"PHASE":"MEN","ENERGY":"Low","STRESS":"Low","MOOD":"Low","CRAMP":"Mild","HEADACHE":0},
        "assertions": [
            {"type":"top_tag",     "domain":"exercise",   "category":"recovery",  "tag":"cramps_relief"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"iron"},
            {"type":"not_top_tag", "domain":"mental",     "category":"guidance",  "tag":"goal_setting"},
            {"type":"score_gap",   "domain":"exercise",   "cat_high":"recovery",  "cat_low":"training", "min_gap":5},
        ],
    },

    # 2. FOL baseline: strength, creatine, goals
    {
        "name": "FOL_BASELINE",
        "inputs": {"PHASE":"FOL","ENERGY":"High","STRESS":"Low","MOOD":"High","CRAMP":"None","HEADACHE":0},
        "assertions": [
            {"type":"top_tag",     "domain":"exercise",   "category":"training",  "tag":"strength"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"creatine"},
            {"type":"top_tag",     "domain":"mental",     "category":"guidance",  "tag":"goal_setting"},
        ],
    },

    # 3. OVU baseline: heavy lifts, social, antioxidants
    {
        "name": "OVU_BASELINE",
        "inputs": {"PHASE":"OVU","ENERGY":"Very High","STRESS":"Low","MOOD":"High","CRAMP":"None","HEADACHE":0},
        "assertions": [
            {"type":"top_tag",     "domain":"exercise",   "category":"training",  "tag":"heavy_strength"},
            {"type":"top_tag",     "domain":"mental",     "category":"guidance",  "tag":"social"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"antioxidants"},
        ],
    },

    # 4. LUT baseline: taper, journaling, anti-inflammatory nutrition
    {
        "name": "LUT_BASELINE",
        "inputs": {"PHASE":"LUT","ENERGY":"Medium","STRESS":"Moderate","MOOD":"Medium","CRAMP":"None","HEADACHE":0},
        "assertions": [
            {"type":"score_gap",   "domain":"exercise",   "cat_high":"recovery",  "cat_low":"training", "min_gap":3},
            {"type":"top_tag",     "domain":"mental",     "category":"guidance",  "tag":"journaling"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"anti_inflammatory"},
        ],
    },

    # 5. LL baseline: PMS mood support, magnesium, exercise taper
    {
        "name": "LL_BASELINE",
        "inputs": {"PHASE":"LL","ENERGY":"Low","STRESS":"High","MOOD":"Low","CRAMP":"Mild","HEADACHE":0},
        "assertions": [
            {"type":"top_tag",     "domain":"mental",     "category":"guidance",  "tag":"mood_support"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"magnesium"},
            {"type":"score_gap",   "domain":"exercise",   "cat_high":"recovery",  "cat_low":"training", "min_gap":5},
        ],
    },

    # 6. OVU + Severe cramps: social blocked, iron blocked, recovery leads
    {
        "name": "OVU_SEVERE_CRAMPS",
        "inputs": {"PHASE":"OVU","ENERGY":"Very High","STRESS":"Low","MOOD":"Medium","CRAMP":"Severe","HEADACHE":0},
        "assertions": [
            {"type":"blocked_tag", "domain":"mental",     "category":"guidance",  "tag":"social"},
            {"type":"blocked_tag", "domain":"nutrition",  "category":"recovery",  "tag":"iron"},
            {"type":"score_gap",   "domain":"exercise",   "cat_high":"recovery",  "cat_low":"training", "min_gap":10},
            {"type":"top_tag",     "domain":"nutrition",  "category":"recovery",  "tag":"anti_inflammatory"},
        ],
    },

    # 7. OVU + Headache 8: heavy strength blocked, social blocked
    {
        "name": "OVU_HEADACHE_8",
        "inputs": {"PHASE":"OVU","ENERGY":"High","STRESS":"Low","MOOD":"High","CRAMP":"None","HEADACHE":8},
        "assertions": [
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"heavy_strength"},
            {"type":"blocked_tag", "domain":"mental",     "category":"guidance",  "tag":"social"},
            {"type":"not_top_tag", "domain":"mental",     "category":"guidance",  "tag":"social"},
        ],
    },

    # 8. MEN + Very High energy: phase still routes to recovery
    {
        "name": "MEN_HIGH_ENERGY",
        "inputs": {"PHASE":"MEN","ENERGY":"Very High","STRESS":"Low","MOOD":"High","CRAMP":"None","HEADACHE":0},
        "assertions": [
            {"type":"not_top_tag", "domain":"exercise",   "category":"training",  "tag":"heavy_strength"},
            {"type":"score_gap",   "domain":"exercise",   "cat_high":"recovery",  "cat_low":"training", "min_gap":3},
        ],
    },

    # 9. FOL + Very Low energy: recovery beats strength
    {
        "name": "FOL_VERY_LOW_ENERGY",
        "inputs": {"PHASE":"FOL","ENERGY":"Very Low","STRESS":"Low","MOOD":"Medium","CRAMP":"None","HEADACHE":0},
        "assertions": [
            {"type":"score_gap",   "domain":"exercise",   "cat_high":"recovery",  "cat_low":"training", "min_gap":5},
            {"type":"top_tag",     "domain":"exercise",   "category":"recovery",  "tag":"low_intensity"},
        ],
    },

    # 10. OVU + Severe stress: high intensity blocked
    {
        "name": "OVU_SEVERE_STRESS",
        "inputs": {"PHASE":"OVU","ENERGY":"High","STRESS":"Severe","MOOD":"Medium","CRAMP":"None","HEADACHE":0},
        "assertions": [
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"high_intensity"},
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"heavy_strength"},
            {"type":"top_tag",     "domain":"mental",     "category":"recovery",  "tag":"breathwork"},
        ],
    },

    # 11. LUT + High energy: taper persists
    {
        "name": "LUT_HIGH_ENERGY",
        "inputs": {"PHASE":"LUT","ENERGY":"High","STRESS":"Low","MOOD":"High","CRAMP":"None","HEADACHE":0},
        "assertions": [
            {"type":"not_top_tag", "domain":"exercise",   "category":"training",  "tag":"hiit"},
            {"type":"not_top_tag", "domain":"exercise",   "category":"training",  "tag":"sprint"},
        ],
    },

    # 12. MEN + Very Severe cramps: intense exercise blocked, anti-inflam leads
    {
        "name": "MEN_VERY_SEVERE_CRAMPS",
        "inputs": {"PHASE":"MEN","ENERGY":"Low","STRESS":"Low","MOOD":"Low","CRAMP":"Very Severe","HEADACHE":0},
        "assertions": [
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"high_intensity"},
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"moderate_intensity"},
            {"type":"top_tag",     "domain":"exercise",   "category":"recovery",  "tag":"cramps_relief"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"anti_inflammatory"},
        ],
    },

    # 13. LL + Severe full-PMS state
    {
        "name": "LL_FULL_PMS",
        "inputs": {"PHASE":"LL","ENERGY":"Very Low","STRESS":"Severe","MOOD":"Very Low","CRAMP":"Severe","HEADACHE":0},
        "assertions": [
            {"type":"blocked_tag", "domain":"mental",     "category":"guidance",  "tag":"social"},
            {"type":"top_tag",     "domain":"mental",     "category":"guidance",  "tag":"mood_support"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"magnesium"},
            {"type":"score_gap",   "domain":"exercise",   "cat_high":"recovery",  "cat_low":"training", "min_gap":10},
        ],
    },

    # 14. Headache 9: most exercise blocked
    {
        "name": "HEADACHE_9",
        "inputs": {"PHASE":"OVU","ENERGY":"High","STRESS":"Low","MOOD":"High","CRAMP":"None","HEADACHE":9},
        "assertions": [
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"high_intensity"},
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"hiit"},
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"heavy_strength"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"hydration"},
        ],
    },

    # 15. FOL + Severe stress: high intensity blocked
    {
        "name": "FOL_SEVERE_STRESS",
        "inputs": {"PHASE":"FOL","ENERGY":"Very High","STRESS":"Severe","MOOD":"High","CRAMP":"None","HEADACHE":0},
        "assertions": [
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"high_intensity"},
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"heavy_strength"},
            {"type":"top_tag",     "domain":"mental",     "category":"recovery",  "tag":"breathwork"},
        ],
    },

    # 16. OVU + Very Severe cramps: maximum restriction
    {
        "name": "OVU_VERY_SEVERE_CRAMPS",
        "inputs": {"PHASE":"OVU","ENERGY":"High","STRESS":"Low","MOOD":"Medium","CRAMP":"Very Severe","HEADACHE":0},
        "assertions": [
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"high_intensity"},
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"moderate_intensity"},
            {"type":"blocked_tag", "domain":"mental",     "category":"guidance",  "tag":"social"},
            {"type":"blocked_tag", "domain":"nutrition",  "category":"recovery",  "tag":"iron"},
        ],
    },

    # 17. MEN + Severe cramps: anti-inflammatory nutrition leads
    {
        "name": "MEN_SEVERE_CRAMPS",
        "inputs": {"PHASE":"MEN","ENERGY":"Low","STRESS":"Moderate","MOOD":"Low","CRAMP":"Severe","HEADACHE":0},
        "assertions": [
            {"type":"top_tag",     "domain":"exercise",   "category":"recovery",  "tag":"cramps_relief"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"anti_inflammatory"},
            {"type":"not_top_tag", "domain":"mental",     "category":"guidance",  "tag":"social"},
        ],
    },

    # 18. LUT + Very Low mood: inward mental routing
    {
        "name": "LUT_LOW_MOOD",
        "inputs": {"PHASE":"LUT","ENERGY":"Medium","STRESS":"Moderate","MOOD":"Very Low","CRAMP":"None","HEADACHE":0},
        "assertions": [
            {"type":"top_tag",     "domain":"mental",     "category":"guidance",  "tag":"journaling"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"mood_support"},
            {"type":"not_top_tag", "domain":"mental",     "category":"guidance",  "tag":"social"},
        ],
    },

    # 19. FOL peak: best-case performance scenario
    {
        "name": "FOL_PEAK",
        "inputs": {"PHASE":"FOL","ENERGY":"Very High","STRESS":"Low","MOOD":"Very High","CRAMP":"None","HEADACHE":0},
        "assertions": [
            {"type":"top_tag",     "domain":"exercise",   "category":"training",  "tag":"heavy_strength"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"creatine"},
            {"type":"top_tag",     "domain":"mental",     "category":"guidance",  "tag":"goal_setting"},
        ],
    },

    # 20. OVU + Moderate cramps: social still encouraged, anti-inflam leads
    {
        "name": "OVU_MODERATE_CRAMPS",
        "inputs": {"PHASE":"OVU","ENERGY":"High","STRESS":"Low","MOOD":"High","CRAMP":"Moderate","HEADACHE":0},
        "assertions": [
            {"type":"top_tag",     "domain":"mental",     "category":"guidance",  "tag":"social"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"recovery",  "tag":"anti_inflammatory"},
            {"type":"top_tag",     "domain":"exercise",   "category":"recovery",  "tag":"cramps_relief"},
        ],
    },

    # 21. LUT + Severe stress: double allostatic load
    {
        "name": "LUT_SEVERE_STRESS",
        "inputs": {"PHASE":"LUT","ENERGY":"Medium","STRESS":"Severe","MOOD":"Low","CRAMP":"None","HEADACHE":0},
        "assertions": [
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"high_intensity"},
            {"type":"top_tag",     "domain":"mental",     "category":"recovery",  "tag":"breathwork"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"anti_inflammatory"},
        ],
    },

    # 22. MEN + Very Severe cramps + Headache 7
    {
        "name": "MEN_WORST_CASE",
        "inputs": {"PHASE":"MEN","ENERGY":"Very Low","STRESS":"High","MOOD":"Very Low","CRAMP":"Very Severe","HEADACHE":7},
        "assertions": [
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"high_intensity"},
            {"type":"blocked_tag", "domain":"exercise",   "category":"training",  "tag":"moderate_intensity"},
            {"type":"blocked_tag", "domain":"mental",     "category":"guidance",  "tag":"social"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"anti_inflammatory"},
        ],
    },

    # 23. FOL + Low mood: unusual conflict
    {
        "name": "FOL_LOW_MOOD",
        "inputs": {"PHASE":"FOL","ENERGY":"Medium","STRESS":"Low","MOOD":"Very Low","CRAMP":"None","HEADACHE":0},
        "assertions": [
            {"type":"top_tag",     "domain":"mental",     "category":"guidance",  "tag":"mood_support"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"mood_support"},
        ],
    },

    # 24. OVU + Headache 5: social suppressed (delta), not blocked
    {
        "name": "OVU_HEADACHE_5",
        "inputs": {"PHASE":"OVU","ENERGY":"High","STRESS":"Low","MOOD":"High","CRAMP":"None","HEADACHE":5},
        "assertions": [
            {"type":"not_top_tag", "domain":"mental",     "category":"guidance",  "tag":"social"},
            {"type":"top_tag",     "domain":"nutrition",  "category":"guidance",  "tag":"hydration"},
        ],
    },

    # 25. LL + Low mood compound: maximum calming stack
    {
        "name": "LL_LOW_MOOD_COMPOUND",
        "inputs": {"PHASE":"LL","ENERGY":"Low","STRESS":"High","MOOD":"Low","CRAMP":"Mild","HEADACHE":0},
        "assertions": [
            {"type":"top_tag",     "domain":"mental",     "category":"guidance",  "tag":"mood_support"},
            {"type":"top_tag",     "domain":"mental",     "category":"recovery",  "tag":"breathwork"},
            {"type":"top_tag",     "domain":"sleep",      "category":"recovery",  "tag":"breathwork"},
        ],
    },
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run():
    passed = 0
    failed = 0
    total  = len(SCENARIOS)

    print(f"\nInBalance Validation Matrix — {total} scenarios\n{'─'*60}")

    for scenario in SCENARIOS:
        results  = recommend(scenario["inputs"], top_n=FETCH_N)
        failures = check(scenario, results)

        if failures:
            failed += 1
            print(f"❌  {scenario['name']}")
            for f in failures:
                print(f)
        else:
            passed += 1
            print(f"✅  {scenario['name']}")

    print(f"\n{'─'*60}")
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED ← review rules")
    else:
        print("  ✓  all clear")
    print()

    return failed == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
