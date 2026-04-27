#!/usr/bin/env python3
"""
Tests for the phase-aware / phase-blind recommender modes.

Locks down the invariants that make the positioning pivot safe:

  1. ``phase_blind`` never touches PHASE-only rules.
  2. ``phase_blind`` strips PHASE sub-conditions from mixed rules so the
     symptom portion still fires.
  3. ``phase_blind`` produces non-empty, sensible recommendations for a
     symptomatic user who has no cycle context.
  4. ``suggest_phase_mode`` picks ``phase_blind`` for the exact fusion
     modes that fusion.py uses as "no phase signal" fallbacks.
  5. The default mode is ``phase_aware`` (backward compatible).
  6. Invalid phase_mode values raise ValueError.

Run:
    python scripts/test_phase_modes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recommender import (
    recommend,
    suggest_phase_mode,
    build_concept_activations,
    check_guards,
    PHASE_AWARE,
    PHASE_BLIND,
)
from recommender.engine import _rule_is_phase_only
from data.rules import RULES


passed = 0
failed = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label}" + (f"  — {detail}" if detail else ""))
    if ok:
        passed += 1
    else:
        failed += 1


# ── Fixtures ──────────────────────────────────────────────────────────────────

SYMPTOMATIC_USER = {
    "ENERGY":        "Low",
    "STRESS":        "High",
    "CRAMP":         "Moderate",
    "MOOD":          "Low",
    "SLEEP_QUALITY": "Poor",
    "BRAIN_FOG":     "Moderate",
    "HEADACHE":      3,
    "BLOATING":      "Low",
    "MOOD_SWING":    "Low",
    "CRAVINGS":      "None",
    "FLOW":          "None",
    "IRRITABILITY":  "Low",
    "BREAST_TENDERNESS": "None",
    "INDIGESTION":   "None",
}

WITH_PHASE_LUT = {**SYMPTOMATIC_USER, "PHASE": "LUT"}


# ── [1] PHASE-only rules never fire in blind mode ─────────────────────────────
print("\n[1] PHASE-only rules never fire in phase_blind mode")

phase_only_rules = [r for r in RULES if _rule_is_phase_only(r)]
phase_only_ids = {r["id"] for r in phase_only_rules}
check(
    "phase-only ruleset is non-empty (sanity)",
    len(phase_only_rules) > 0,
    f"found {len(phase_only_rules)} PHASE-only rules",
)

# Feed a user with every PHASE value plausible — if any PHASE-only rule sneaks
# through in blind mode, the audit trail would mention its id.
for phase_val in ["MEN", "FOL", "OVU", "LUT", "LL"]:
    inputs = {**SYMPTOMATIC_USER, "PHASE": phase_val}
    _, contributions = build_concept_activations(inputs, phase_mode=PHASE_BLIND)
    fired_ids = set()
    for domain_con in contributions.values():
        for tag_traces in domain_con.values():
            for trace in tag_traces:
                # trace format: "<rule_id>(+X.Y): <rationale>"
                rid = trace.split("(", 1)[0]
                fired_ids.add(rid)
    leaked = fired_ids & phase_only_ids
    check(
        f"PHASE={phase_val}: no PHASE-only rule activations in blind mode",
        not leaked,
        f"leaked: {sorted(leaked)}" if leaked else "",
    )


# ── [2] Mixed rules fire in blind mode when symptoms match ────────────────────
print("\n[2] Mixed (PHASE + symptom) rules fire on symptom portion in blind mode")

# Find mixed rules (have conditions list with both PHASE and non-PHASE)
mixed_rules = []
for r in RULES:
    if "conditions" not in r:
        continue
    sigs = [c.get("signal") for c in r["conditions"]]
    if "PHASE" in sigs and any(s != "PHASE" for s in sigs):
        mixed_rules.append(r)

check(
    "mixed-rule set is non-empty",
    len(mixed_rules) > 0,
    f"found {len(mixed_rules)} mixed rules",
)

if mixed_rules:
    # Pick a mixed rule and craft an input that satisfies the non-PHASE
    # conditions but has a DIFFERENT phase than the one the rule expects —
    # in blind mode it should still fire; in aware mode it should not.
    # We pick the first mixed rule whose non-PHASE conditions are simple.
    def _reverse_phase(p: str) -> str:
        return {"MEN": "LUT", "FOL": "MEN", "OVU": "MEN", "LUT": "FOL", "LL": "FOL"}.get(p, "MEN")

    target = None
    for r in mixed_rules:
        phase_cond = next(c for c in r["conditions"] if c.get("signal") == "PHASE")
        target = (r, phase_cond["value"])
        break

    if target is not None:
        rule, rule_phase = target
        wrong_phase = _reverse_phase(rule_phase)
        # Build inputs that satisfy every non-PHASE cond in this rule.
        from recommender.scales import SIGNAL_RANKS
        crafted = dict(SYMPTOMATIC_USER)
        crafted["PHASE"] = wrong_phase
        for cond in rule["conditions"]:
            sig = cond.get("signal")
            if sig == "PHASE":
                continue
            op = cond.get("op")
            val = cond.get("value")
            if sig == "HEADACHE":
                # numeric; pick a value satisfying op
                try:
                    target_num = float(val) + (1 if op in (">=", ">") else -1 if op in ("<=", "<") else 0)
                    crafted[sig] = max(0, int(target_num))
                except ValueError:
                    crafted[sig] = 5
                continue
            ranks = SIGNAL_RANKS.get(sig, {})
            if op == "=" and val in ranks:
                crafted[sig] = val
            elif op in (">=", ">") and ranks:
                # Pick the highest-ranked value (satisfies >=)
                crafted[sig] = max(ranks, key=ranks.get)
            elif op in ("<=", "<") and ranks:
                crafted[sig] = min(ranks, key=ranks.get)
            else:
                crafted[sig] = val

        _, contrib_blind = build_concept_activations(crafted, phase_mode=PHASE_BLIND)
        _, contrib_aware = build_concept_activations(crafted, phase_mode=PHASE_AWARE)

        def _rule_fired(contrib, rid: str) -> bool:
            for d in contrib.values():
                for traces in d.values():
                    for t in traces:
                        if t.startswith(f"{rid}("):
                            return True
            return False

        check(
            f"mixed rule {rule['id']} fires in blind mode despite wrong PHASE",
            _rule_fired(contrib_blind, rule["id"]),
        )
        check(
            f"mixed rule {rule['id']} does NOT fire in aware mode with wrong PHASE",
            not _rule_fired(contrib_aware, rule["id"]),
        )


# ── [3] Blind mode produces usable recommendations without PHASE ──────────────
print("\n[3] phase_blind produces non-empty, sensible recommendations")

rec_blind = recommend(SYMPTOMATIC_USER, top_n=3, phase_mode=PHASE_BLIND)  # no PHASE key
check("rec_blind has all 4 domains", set(rec_blind.keys()) >= {"exercise", "nutrition", "sleep", "mental"})

total_items = sum(len(cats) for d in rec_blind.values() for cats in d.values())
check("rec_blind returns at least 20 items overall", total_items >= 20, f"got {total_items}")

# At least one activity should have a non-zero delta_sum — symptoms should push
# some rules. Otherwise the blind recommender would be pure base-score ordering.
has_nonzero = any(
    item["delta_sum"] != 0.0
    for d in rec_blind.values() for cats in d.values() for item in cats
)
check("rec_blind has non-zero symptom-driven activations", has_nonzero)


# ── [4] suggest_phase_mode on fusion fallbacks ────────────────────────────────
print("\n[4] suggest_phase_mode routes fusion outputs correctly")

check(
    "layer1_only_no_history → phase_blind",
    suggest_phase_mode({"mode": "layer1_only_no_history", "layer1": {}, "layer2": None}) == PHASE_BLIND,
)
check(
    "layer1_only_non_menstrual → phase_blind",
    suggest_phase_mode({"mode": "layer1_only_non_menstrual", "layer1": {}, "layer2": None}) == PHASE_BLIND,
)
check(
    "fused_non_menstrual + high confidence + strong L2 → phase_aware",
    suggest_phase_mode({
        "mode": "fused_non_menstrual",
        "layer1": {"cycle_day": 12, "forecast_confidence": "high"},
        "layer2": {"top_prob": 0.75},
    }) == PHASE_AWARE,
)
check(
    "fused_non_menstrual + low forecast confidence → phase_blind",
    suggest_phase_mode({
        "mode": "fused_non_menstrual",
        "layer1": {"cycle_day": 12, "forecast_confidence": "low"},
        "layer2": {"top_prob": 0.75},
    }) == PHASE_BLIND,
)
check(
    "fused_non_menstrual + weak L2 top_prob → phase_blind",
    suggest_phase_mode({
        "mode": "fused_non_menstrual",
        "layer1": {"cycle_day": 12, "forecast_confidence": "high"},
        "layer2": {"top_prob": 0.25},
    }) == PHASE_BLIND,
)
check(
    "keyword override: has_period_history=False → phase_blind",
    suggest_phase_mode(has_period_history=False) == PHASE_BLIND,
)


# ── [5] Default is phase_aware (backward compatibility) ───────────────────────
print("\n[5] Default mode is phase_aware")

rec_default   = recommend(WITH_PHASE_LUT, top_n=3)
rec_explicit  = recommend(WITH_PHASE_LUT, top_n=3, phase_mode=PHASE_AWARE)

default_ids  = {(d, c, i["id"]) for d, cats in rec_default.items()  for c, items in cats.items() for i in items}
explicit_ids = {(d, c, i["id"]) for d, cats in rec_explicit.items() for c, items in cats.items() for i in items}
check("default recommend() == phase_aware recommend()", default_ids == explicit_ids)


# ── [6] Invalid phase_mode raises ─────────────────────────────────────────────
print("\n[6] Invalid phase_mode raises ValueError")

try:
    recommend(SYMPTOMATIC_USER, phase_mode="aggressive_periodization")
except ValueError:
    check("ValueError raised for unknown phase_mode", True)
else:
    check("ValueError raised for unknown phase_mode", False, "no error raised")


# ── [7] check_guards also honours phase_blind ─────────────────────────────────
print("\n[7] check_guards skips PHASE-only BLOCK rules in phase_blind")

# Find a PHASE-only BLOCK rule if any exists.
phase_only_blocks = [r for r in RULES if r.get("action") == "BLOCK" and _rule_is_phase_only(r)]
if phase_only_blocks:
    # Pick an activity in the same domain with tags the block targets.
    from data.activities import ACTIVITIES
    block = phase_only_blocks[0]
    target_acts = [a for a in ACTIVITIES if a["domain"] == block["domain"]
                   and set(block.get("tags", [])) & set(a.get("tags", []))]
    if target_acts:
        activity = target_acts[0]
        # Inputs that trigger the block in aware mode.
        phase_val = block["value"] if "value" in block else block["conditions"][0]["value"]
        aware_inputs = {**SYMPTOMATIC_USER, "PHASE": phase_val}
        is_blocked_aware, _ = check_guards(activity, aware_inputs, phase_mode=PHASE_AWARE)
        is_blocked_blind, _ = check_guards(activity, aware_inputs, phase_mode=PHASE_BLIND)
        check("PHASE-only block fires in aware mode",       is_blocked_aware,  f"rule={block['id']}")
        check("PHASE-only block does NOT fire in blind mode", not is_blocked_blind, f"rule={block['id']}")
    else:
        check("no target activity for phase-only block test", True, "skipped — no matching activity")
else:
    check("no phase-only BLOCK rules in dataset", True, "skipped")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed.")
    sys.exit(0)
else:
    print("FAILURES.")
    sys.exit(1)
