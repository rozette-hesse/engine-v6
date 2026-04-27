"""
InBalance Graph Recommender
===========================
Scores and ranks activities using a spreading-activation graph built from
DELTA rules.  BLOCK rules remain hard guards — they are not part of the graph.

Pipeline
--------
  1. ``build_concept_activations``  — walk every DELTA rule; for each that fires,
     spread delta / min(n_tags, 4) evenly across its tag-concept nodes.
  2. ``spread``                     — score each activity by summing activations
     over its tag set within its domain.
  3. ``check_guards``               — evaluate BLOCK rules; blocked items are
     sorted to the bottom but still returned for transparency.
  4. Rank within domain × category; return top_n per group.

Normalization
-------------
  Per-node delta = rule_delta / min(n_tags, 4).
  The cap of 4 keeps broad rules (n > 4) effective while dampening
  double-counting for small, focused rules (n ≤ 4).

Phase modes
-----------
  ``phase_aware`` (default)
      Current behaviour — every rule's PHASE condition is evaluated against
      inputs['PHASE']. PHASE-only rules and mixed (PHASE + symptom) rules
      both require a PHASE match to fire.

  ``phase_blind``
      Drops the PHASE filter from every rule condition. PHASE-only rules
      (53 of 171, ~31% of the ruleset — mostly athletic periodization) are
      skipped entirely. Mixed rules (10 of 171) keep their non-PHASE
      conditions and fire on the symptom portion alone. Used when the
      cycle engine is low-confidence or the user has no period history.

  See ``suggest_phase_mode(fusion)`` for a helper that picks automatically
  from a fusion output's confidence signals.

Public interface
----------------
  recommend(inputs, top_n=3, phase_mode="phase_aware")
      → dict[domain, dict[category, list[item]]]
  suggest_phase_mode(fusion) → "phase_aware" | "phase_blind"
"""

from collections import defaultdict
from typing import Dict, Optional

from data.activities import ACTIVITIES
from data.rules import RULES
from .scales import SIGNAL_RANKS


# Phase mode constants — exported for type-ish usage by callers.
PHASE_AWARE = "phase_aware"
PHASE_BLIND = "phase_blind"
_VALID_PHASE_MODES = {PHASE_AWARE, PHASE_BLIND}


# ── Signal matching ───────────────────────────────────────────────────────────

def _get_input_value(signal: str, inputs: dict):
    """Return (numeric_rank, raw_value) for a given signal key."""
    raw = inputs.get(signal)
    if raw is None:
        return None, None
    ranks = SIGNAL_RANKS.get(signal)
    if ranks:
        return ranks.get(str(raw), None), str(raw)
    try:
        return float(raw), raw
    except (TypeError, ValueError):
        return None, raw


def _compare(a, op: str, b) -> bool:
    ops = {"=": a == b, "!=": a != b, ">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b}
    return ops.get(op, False)


def _signal_matches(rule: dict, inputs: dict) -> bool:
    """Return True if a single-condition rule fires against the given inputs."""
    signal = rule["signal"]
    op     = rule["op"]
    val    = rule["value"]

    if signal == "PHASE":
        return op == "=" and inputs.get("PHASE") == val

    if signal == "HEADACHE":
        try:
            return _compare(float(inputs.get("HEADACHE", 0)), op, float(val))
        except (TypeError, ValueError):
            return False

    rank_val, _ = _get_input_value(signal, inputs)
    if rank_val is None:
        return False

    ranks = SIGNAL_RANKS[signal]
    try:
        threshold = ranks[val]
    except KeyError:
        try:
            threshold = float(val)
        except ValueError:
            return False

    return _compare(rank_val, op, threshold)


def _rule_is_phase_only(rule: dict) -> bool:
    """True if every condition on this rule targets PHASE."""
    if "conditions" in rule:
        return bool(rule["conditions"]) and all(
            c.get("signal") == "PHASE" for c in rule["conditions"]
        )
    return rule.get("signal") == "PHASE"


def _rule_condition_matches(
    rule: dict,
    inputs: dict,
    phase_mode: str = PHASE_AWARE,
) -> bool:
    """
    Evaluate rule conditions against inputs.

    Multi-condition rules use a ``conditions`` list; all must match (AND logic).
    Single-condition rules use top-level signal/op/value fields.

    In ``phase_blind`` mode, PHASE sub-conditions are stripped from the
    AND chain — a mixed PHASE+symptom rule still fires on its symptom
    portion alone. PHASE-only rules return False (short-circuited upstream
    in ``build_concept_activations`` / ``check_guards`` so they never even
    reach this function).
    """
    if phase_mode == PHASE_BLIND and _rule_is_phase_only(rule):
        # Defensive — callers are expected to filter these out first.
        return False

    if "conditions" in rule:
        conds = rule["conditions"]
        if phase_mode == PHASE_BLIND:
            conds = [c for c in conds if c.get("signal") != "PHASE"]
            if not conds:
                # Rule had only PHASE conditions — don't fire in blind mode.
                return False
        return all(_signal_matches(cond, inputs) for cond in conds)
    return _signal_matches(rule, inputs)


def _tags_match(rule: dict, activity: dict) -> bool:
    """Check whether a rule's tag set matches an activity (ANY or ALL mode)."""
    rule_tags = rule.get("tags", [])
    act_tags  = set(activity.get("tags", []))
    if not rule_tags:
        return True
    if rule["match_mode"] == "ALL":
        return all(t in act_tags for t in rule_tags)
    return any(t in act_tags for t in rule_tags)  # ANY


# ── Graph layer: concept activation ──────────────────────────────────────────

def build_concept_activations(
    inputs: dict,
    phase_mode: str = PHASE_AWARE,
) -> tuple:
    """
    Walk all DELTA rules. For each that fires, distribute delta evenly across
    its concept nodes (tags), scoped to the rule's domain.

    In ``phase_blind`` mode, PHASE-only rules are skipped entirely and mixed
    rules are evaluated with their PHASE conditions stripped.

    Returns
    -------
    activations   : {domain: {tag: float}}   net activation per concept node
    contributions : {domain: {tag: [str]}}   audit trail for path tracing
    """
    activations   = defaultdict(lambda: defaultdict(float))
    contributions = defaultdict(lambda: defaultdict(list))

    for rule in RULES:
        if rule["action"] != "DELTA":
            continue
        tags  = rule.get("tags") or []
        delta = rule.get("delta") or 0
        if not tags or delta == 0:
            continue
        if phase_mode == PHASE_BLIND and _rule_is_phase_only(rule):
            continue
        if not _rule_condition_matches(rule, inputs, phase_mode=phase_mode):
            continue

        domain   = rule["domain"]
        # Cap divisor at 4: broad rules stay effective; small rules don't
        # earn extra credit for multi-tag activities.
        per_node = delta / min(len(tags), 4)

        for tag in tags:
            activations[domain][tag]   += per_node
            contributions[domain][tag].append(
                f"{rule['id']}({per_node:+.1f}): {rule['rationale']}"
            )

    return dict(activations), dict(contributions)


# ── Graph layer: spreading activation (activity scoring) ─────────────────────

def spread(activity: dict, activations: dict, contributions: dict) -> tuple:
    """
    Score an activity by summing concept activations across all its tags.

    Returns
    -------
    delta_sum    : float   total activation received
    active_paths : [str]   human-readable concept → rule traces (for auditing)
    """
    domain     = activity["domain"]
    domain_act = activations.get(domain, {})
    domain_con = contributions.get(domain, {})

    delta_sum    = 0.0
    active_paths = []

    for tag in activity.get("tags", []):
        w = domain_act.get(tag, 0.0)
        if w != 0.0:
            delta_sum += w
            for trace in domain_con.get(tag, []):
                active_paths.append(f"[{tag}] {trace}")

    return delta_sum, active_paths


# ── Guards ────────────────────────────────────────────────────────────────────

def check_guards(
    activity: dict,
    inputs: dict,
    phase_mode: str = PHASE_AWARE,
) -> tuple:
    """
    Evaluate all BLOCK rules against an activity.
    Blocks are absolute — they are not part of the graph traversal.

    In ``phase_blind`` mode, PHASE-only BLOCK rules are skipped. This is
    the correct behaviour — when we don't know the user's phase, a
    phase-contingent block can't be applied. Mixed-condition blocks still
    fire on their non-PHASE conditions (conservatively still blocking).

    Returns
    -------
    is_blocked    : bool
    guard_reasons : [str]  which rules blocked and why
    """
    is_blocked    = False
    guard_reasons = []
    domain        = activity["domain"]

    for rule in RULES:
        if rule["action"] != "BLOCK":
            continue
        if rule["domain"] != domain:
            continue
        if phase_mode == PHASE_BLIND and _rule_is_phase_only(rule):
            continue
        if not _rule_condition_matches(rule, inputs, phase_mode=phase_mode):
            continue
        if not _tags_match(rule, activity):
            continue
        is_blocked = True
        guard_reasons.append(f"{rule['id']}: {rule['rationale']}")

    return is_blocked, guard_reasons


# ── Main entry point ──────────────────────────────────────────────────────────

def recommend(
    inputs: dict,
    top_n: int = 3,
    phase_mode: str = PHASE_AWARE,
) -> dict:
    """
    Score and rank all activities for the given user inputs.

    Parameters
    ----------
    inputs      : dict with keys PHASE, ENERGY, STRESS, MOOD, CRAMP, HEADACHE, etc.
    top_n       : items to return per domain × category group
    phase_mode  : ``"phase_aware"`` (default) or ``"phase_blind"``.
                  In blind mode, PHASE-only rules are skipped and PHASE
                  conditions are stripped from mixed rules. Use when the
                  cycle engine is low-confidence or the user has no period
                  history. See ``suggest_phase_mode()``.

    Returns
    -------
    dict[domain][category] → list of scored activity dicts, ranked 1..top_n.
    Blocked items are sorted to the bottom but included for transparency.
    """
    if phase_mode not in _VALID_PHASE_MODES:
        raise ValueError(
            f"phase_mode must be one of {_VALID_PHASE_MODES}, got {phase_mode!r}"
        )

    # Step 1: build concept activations in a single pass over RULES
    activations, contributions = build_concept_activations(inputs, phase_mode=phase_mode)

    results = []
    for activity in ACTIVITIES:
        # Step 2: guard check (hard blocks, order-independent)
        is_blocked, guard_reasons = check_guards(activity, inputs, phase_mode=phase_mode)

        # Step 3: spreading activation (O(tags_per_activity))
        delta_sum, active_paths = spread(activity, activations, contributions)

        results.append({
            "id":             activity["id"],
            "domain":         activity["domain"],
            "category":       activity["category"],
            "text":           activity["text"],
            "base_score":     activity["base_score"],
            "delta_sum":      round(delta_sum, 2),
            "final_score":    round(activity["base_score"] + delta_sum, 2),
            "intensity":      activity.get("intensity", ""),
            "dur_min":        activity.get("dur_min", 0),
            "dur_max":        activity.get("dur_max", 0),
            "confidence":     activity.get("confidence", 0.8),
            "is_blocked":     is_blocked,
            "fired_concepts": active_paths,
            "guard_reasons":  guard_reasons,
            "tags":           list(set(activity.get("tags", []))),
        })

    # Step 4: rank within domain × category, return top_n per group
    groups = defaultdict(list)
    for r in results:
        groups[(r["domain"], r["category"])].append(r)

    output = defaultdict(lambda: defaultdict(list))
    for (domain, category), items in groups.items():
        items.sort(key=lambda x: (x["is_blocked"], -x["final_score"]))
        for rank, item in enumerate(items[:top_n], 1):
            item["rank"] = rank
            output[domain][category].append(item)

    return dict(output)


# ── Phase-mode selection helper ───────────────────────────────────────────────

def suggest_phase_mode(
    fusion: Optional[Dict[str, object]] = None,
    *,
    has_period_history: Optional[bool] = None,
    forecast_confidence: Optional[str] = None,
    min_layer2_prob: float = 0.40,
) -> str:
    """
    Pick the appropriate recommender phase mode from a fusion output.

    Conservative default: return ``phase_aware`` only when we have real
    reasons to trust the phase. Otherwise fall back to ``phase_blind`` so
    the recommender still produces good symptom-driven output instead of
    silently misapplying periodization rules.

    Accepts either a full fusion dict (from ``cycle.get_fused_output``) or
    the individual signals as keyword arguments — handy for callers that
    already pulled the fields out.

    Returns ``"phase_aware"`` or ``"phase_blind"``.
    """
    # Derive signals from the fusion dict when provided.
    if fusion is not None:
        mode = fusion.get("mode")
        # Explicit low-info modes → blind, no matter what else.
        if mode in ("layer1_only_no_history", "layer1_only_non_menstrual"):
            return PHASE_BLIND

        layer1 = fusion.get("layer1") or {}
        layer2 = fusion.get("layer2") or {}
        if has_period_history is None:
            # If fusion has a non-None layer1 with a known cycle_day, we had
            # period history. This mirrors the gates in fusion.get_fused_output.
            has_period_history = bool(layer1.get("cycle_day"))
        if forecast_confidence is None:
            forecast_confidence = str(layer1.get("forecast_confidence", "high"))
        layer2_top_prob = float((layer2 or {}).get("top_prob") or 0.0)
    else:
        layer2_top_prob = 1.0  # not provided → don't veto on this signal

    if has_period_history is False:
        return PHASE_BLIND
    if forecast_confidence == "low":
        return PHASE_BLIND
    if layer2_top_prob < min_layer2_prob:
        return PHASE_BLIND

    return PHASE_AWARE
