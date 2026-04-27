"""
Cycle Fusion Layer
==================
Orchestrates the three prediction layers into a single output dict.

Decision tree
-------------
  period_start_logged = True
      → mode: period_start_override, phase: Menstrual

  cycle_day ≤ 7 AND flow evidence meets per-day thresholds
      → mode: menstrual_ongoing, phase: Menstrual

  no period history
      → mode: layer1_only_no_history, phase from L1 3-class prior

  no symptom input
      → mode: layer1_only_non_menstrual, phase from L1 3-class prior

  otherwise
      → mode: fused_non_menstrual
        L1 + L2 fused via per-phase convex combination
        Constrained to plausible phases based on L1 baseline + L2 confidence
        Layer 3 produces the timing annotation

Fusion weights
--------------
  v3/v4: L1 weight = 0.10 baseline (L2 is cycle-day aware, so L1 adds less).
  v1/v2: L1 weight = 0.20 baseline.
  Both versions boost L1 weight slightly when L1 strongly signals Fertility
  or Follicular to avoid over-weighting a confused L2.
"""

import logging
from typing import Dict, List, Optional

from .config import (
    LAYER1_WEIGHT,
    LAYER1_WEIGHT_V3,
    LAYER2_WEIGHT,
    LAYER2_WEIGHT_V3,
    MODEL_VERSION,
    NON_MENSTRUAL_PHASES,
)
from .layer1_period import get_layer1_output
from .layer2 import get_layer2_v1_output, get_layer2_v3_output, get_layer2_v4_output, get_layer2_v5_output
from .layer2.base import MUCUS_FERTILITY_MAP, _normalize_mucus
from .layer3_timing import get_layer3_output
from .utils import normalize_probs


# Dedicated logger for personalization flip events. Off by default (NullHandler).
# Callers / app config can attach a handler to ingest these events for audit:
#   logging.getLogger("cycle.personalization.flip").addHandler(<handler>)
# Each event records before/after top phases, the blend weight, and whether
# personalization flipped the argmax. Messages are emitted as structured
# `extra` fields so JSON log handlers pick them up cleanly.
_flip_logger = logging.getLogger("cycle.personalization.flip")
_flip_logger.addHandler(logging.NullHandler())


# ── Flow categorisation ───────────────────────────────────────────────────────
# Used by the menstrual-continuation detector.

_SUBSTANTIAL_FLOWS = {
    "heavy", "very heavy", "moderate",
    "somewhat heavy", "somewhat light", "light",
}
_SPOTTING_FLOWS = {
    "spotting", "spotting / very light", "spotting/very light", "very light",
}


def _flow_category(flow: Optional[str]) -> str:
    """Bucket a flow string into: missing | none | spotting | substantial."""
    if flow is None:
        return "missing"
    f = str(flow).strip().lower()
    if f in {"", "unknown"}:
        return "missing"
    if f in {"none", "not at all", "0"}:
        return "none"
    if f in _SPOTTING_FLOWS:
        return "spotting"
    if f in _SUBSTANTIAL_FLOWS:
        return "substantial"
    # Unknown non-empty label → err on the side of keeping the menstrual window
    return "substantial"


# ── Menstrual continuation detector ──────────────────────────────────────────

def _is_ongoing_menstrual(
    layer1: Dict[str, object],
    flow: Optional[str],
    recent_daily_logs: Optional[List[Dict[str, object]]],
) -> bool:
    """
    Data-driven menstrual-continuation rule calibrated on mcphases_v3.

    Per-cycle-day thresholds were derived by cross-tabbing
    cycle_day × flow_category × true_phase to maximise Menstrual recall
    without creating too many Follicular → Menstrual false positives.

    Dataset uses hormone-based phase labels, so Menstrual can persist after
    visible bleeding stops — "none" flow ≠ Follicular in early cycle days.

    Days 1–4 : Always Menstrual.  Even with "none" flow, day-4 rows are
               Menstrual in the dataset 75% of the time (+18 TP, +6 FP).
    Day 5    : Menstrual unless flow is explicitly spotting or substantial
               AND flow was also absent the day before.  In practice:
               accept none/missing as Menstrual (+21 TP, +11 FP), keep
               spotting/substantial as the only exit signals.
               Calibrated: none-flow day-5 rows are 66% Menstrual in dataset.
    Day 6    : Menstrual only if flow is substantial.
               (77 % of day-6 substantial-flow rows are truly Menstrual.)
    Day 7-8  : Menstrual only if today AND yesterday both had substantial flow.
               Single-day substantial flow at day 7 is ~50/50 — too ambiguous.
               Day 8 follows the same rule: on the held-out set, extending the
               2-day-substantial rule to day 8 flipped 3 Menstrual→Follicular
               errors to correct with 0 regressions (+0.47pp). On the full
               2022 wave: +5 TP, −2 FP (+0.09pp net).
    Day 9+   : Never. Late-cycle Menstrual rows with sustained substantial
               flow are too rare to justify the FP risk.
    """
    cycle_day = layer1.get("cycle_day")
    if not cycle_day:
        return False
    cycle_day = int(cycle_day)
    fc = _flow_category(flow)

    if 1 <= cycle_day <= 4:
        return True
    if cycle_day == 5:
        return fc in {"substantial", "missing", "none"}
    if cycle_day == 6:
        return fc == "substantial"
    if cycle_day in (7, 8):
        if fc != "substantial":
            return False
        for log in (recent_daily_logs or [])[-1:]:
            if _flow_category(log.get("flow")) == "substantial":
                return True
        return False

    return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def has_symptom_input(
    symptoms: Optional[List[str]],
    cervical_mucus: str,
    appetite: int = 0,
    exerciselevel: int = 0,
    recent_daily_logs: Optional[List[Dict[str, object]]] = None,
    symptom_severities: Optional[Dict[str, int]] = None,
) -> bool:
    """Return True if there is any non-default symptom or lifestyle input."""
    has_severities = bool(symptom_severities) and any(
        int(v or 0) > 0 for v in symptom_severities.values()
    )
    return (
        bool(symptoms)
        or has_severities
        or ((cervical_mucus or "unknown").lower() != "unknown")
        or int(appetite) != 0
        or int(exerciselevel) != 0
        or bool(recent_daily_logs)
    )


def _map_layer1_to_non_menstrual(layer1_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Collapse the 4-class L1 distribution to the 3-class non-menstrual space.

    Menstrual probability is added to Follicular (not Luteal) because the
    most likely phase immediately after menstruation is Follicular.
    """
    mapped = {
        "Follicular": float(layer1_probs.get("Follicular", 0.0) + layer1_probs.get("Menstrual", 0.0)),
        "Fertility":  float(layer1_probs.get("Fertility",  0.0)),
        "Luteal":     float(layer1_probs.get("Luteal",     0.0)),
    }
    return normalize_probs(mapped)


def _get_layer1_non_menstrual_top_phase(layer1_probs_3: Dict[str, float]) -> str:
    return max(layer1_probs_3, key=layer1_probs_3.get)


def _get_allowed_phases(baseline_phase: str) -> List[str]:
    """
    Conservative phase reachability: which phases can plausibly follow
    the L1 baseline phase within a single cycle day?
    """
    if baseline_phase == "Follicular":
        return ["Follicular", "Fertility"]
    if baseline_phase == "Fertility":
        return ["Follicular", "Fertility", "Luteal"]
    if baseline_phase == "Luteal":
        return ["Follicular", "Fertility", "Luteal"]
    return NON_MENSTRUAL_PHASES


def _constrain_non_menstrual_probs(
    fused_probs: Dict[str, float],
    baseline_phase: str,
    layer2: Dict[str, object],
    model_version: str = MODEL_VERSION,
    forecast_confidence: str = "high",
) -> Dict[str, float]:
    """
    Constrain fused probabilities to a plausible subset of phases.

    When timing is unreliable (low forecast confidence), open all phases
    so symptom signals can determine the result.

    For v3/v4, the model is cycle-day aware so we trust its top pick more
    aggressively than for v1/v2.
    """
    if forecast_confidence == "low":
        # Timing is unreliable; let symptoms decide freely
        total = sum(fused_probs.values())
        if total <= 0:
            return {p: 1.0 / len(NON_MENSTRUAL_PHASES) for p in NON_MENSTRUAL_PHASES}
        return {k: v / total for k, v in fused_probs.items()}

    allowed = set(_get_allowed_phases(baseline_phase))

    top_prob           = layer2.get("top_prob", 0.0)
    prob_gap           = layer2.get("prob_gap", 0.0)
    signal_confidence  = layer2.get("signal_confidence", "low")
    symptom_phase      = layer2.get("top_phase", baseline_phase)

    if model_version in ("v3", "v4", "v5"):
        l1_l2_agree = (baseline_phase == symptom_phase)
        if l1_l2_agree and (top_prob >= 0.55 or prob_gap >= 0.20):
            # L1 timing and L2 symptoms agree AND L2 is confident → full trust
            allowed = set(NON_MENSTRUAL_PHASES)
        elif not l1_l2_agree and top_prob >= 0.75 and prob_gap >= 0.30:
            # L2 strongly disagrees with timing — add symptom phase but keep
            # L1's baseline as an anchor (don't fully unlock all phases).
            # This prevents a Luteal-biased L2 from overriding clear timing signals.
            allowed.add(symptom_phase)
        elif signal_confidence == "low" and prob_gap < 0.08 and not l1_l2_agree:
            # Weak, ambiguous signal — keep both candidates only
            allowed = {baseline_phase, symptom_phase}
        else:
            allowed.add(symptom_phase)
    else:
        # Legacy v1/v2
        if (
            baseline_phase == "Follicular"
            and symptom_phase == "Luteal"
            and (signal_confidence == "high" or top_prob > 0.65)
        ):
            allowed.add("Luteal")
        if signal_confidence == "low" and prob_gap < 0.10 and baseline_phase != symptom_phase:
            allowed = {baseline_phase, symptom_phase}

    constrained = {p: (v if p in allowed else 0.0) for p, v in fused_probs.items()}
    total       = sum(constrained.values())
    if total <= 0:
        constrained = {p: (1.0 if p == baseline_phase else 0.0) for p in NON_MENSTRUAL_PHASES}
    else:
        constrained = {k: v / total for k, v in constrained.items()}
    return constrained


def _fuse_non_menstrual_probs(
    layer1_probs_3: Dict[str, float],
    layer2_probs_3: Dict[str, float],
    mucus_fertility_score: float = 0.0,
    model_version: str = MODEL_VERSION,
    context_completeness: float = 1.0,
) -> Dict[str, float]:
    """
    Convex combination of L1 and L2 phase probabilities, per-phase.

    Fertility and Follicular phases get a slightly higher L1 weight when L1
    strongly signals them, to compensate for cases where L2 is underconfident.
    A mucus fertility bonus is added to the Fertility phase after fusion.

    v4 strips mucus from its training features so the mucus bonus carries
    more weight at inference time than it did in v3.

    Context completeness
    --------------------
    When the user has not logged recently, L2's history-dependent features
    (bleeding lags, 3-day symptom window) are unreliable.  ``context_completeness``
    is the fraction of the last 3 days that were actually logged (0.0–1.0).
    When incomplete, the base L1 weight is boosted by up to 15 percentage
    points so the calendar signal carries proportionally more weight.
    At full completeness (1.0) behaviour is identical to before.
    """
    # ── Context-aware L1 base weight ─────────────────────────────────────────
    # Boost L1 linearly when recent context is missing.
    # Max boost: +0.15 of the remaining (1 - base) headroom.
    _CONTEXT_BOOST = 0.15
    if model_version in ("v3", "v4", "v5"):
        base_l1_w      = LAYER1_WEIGHT_V3
        mucus_bonus_factor = 0.8 if model_version in ("v4", "v5") else 0.6
    else:
        base_l1_w      = LAYER1_WEIGHT
        mucus_bonus_factor = 1.2

    gap_fraction = 1.0 - max(0.0, min(1.0, context_completeness))
    l1_w         = base_l1_w + gap_fraction * _CONTEXT_BOOST * (1.0 - base_l1_w)

    # ── Per-phase L1 weights (timing-signal boost when L1 is confident) ───────
    if model_version in ("v3", "v4", "v5"):
        fertility_l1  = min(0.30, l1_w + max(0.0, layer1_probs_3.get("Fertility",  0.0) - 0.50))
        follicular_l1 = min(0.30, l1_w + max(0.0, layer1_probs_3.get("Follicular", 0.0) - 0.55))
        luteal_l1     = l1_w
    else:
        fertility_l1  = min(0.60, l1_w + max(0.0, layer1_probs_3.get("Fertility",  0.0) - 0.30))
        follicular_l1 = min(0.60, l1_w + max(0.0, layer1_probs_3.get("Follicular", 0.0) - 0.40))
        luteal_l1     = l1_w

    fused: Dict[str, float] = {}
    for phase in NON_MENSTRUAL_PHASES:
        if phase == "Fertility":
            l1 = fertility_l1
        elif phase == "Follicular":
            l1 = follicular_l1
        else:
            l1 = luteal_l1
        fused[phase] = l1 * layer1_probs_3.get(phase, 0.0) + (1.0 - l1) * layer2_probs_3.get(phase, 0.0)

    # Inference-time mucus bonus (independent of model features)
    fused["Fertility"] = fused.get("Fertility", 0.0) + max(0.0, mucus_fertility_score - 0.50) * mucus_bonus_factor
    return normalize_probs(fused)


# ── Main entry point ──────────────────────────────────────────────────────────

def get_fused_output(
    period_starts: List[str],
    symptoms: Optional[List[str]] = None,
    cervical_mucus: str = "unknown",
    appetite: int = 0,
    exerciselevel: int = 0,
    period_start_logged: bool = False,
    recent_daily_logs: Optional[List[Dict[str, object]]] = None,
    today: Optional[str] = None,
    flow: str = "none",
    *,
    symptom_severities: Optional[Dict[str, int]] = None,
    model_version: Optional[str] = None,
    personal_stats: Optional[Dict] = None,
    context_completeness: float = 1.0,
) -> Dict[str, object]:
    """
    Run the full multi-layer cycle prediction fusion.

    Parameters
    ----------
    period_starts        : list of YYYY-MM-DD period start dates
    symptoms             : binary symptom names (legacy — use symptom_severities)
    cervical_mucus       : observed mucus type
    appetite             : 0-10 appetite level
    exerciselevel        : 0-10 exercise level
    period_start_logged  : user explicitly logged a new period start today
    recent_daily_logs    : list of prior-day observation dicts for 3-day history
    today                : override for today's date (YYYY-MM-DD), for testing
    flow                 : observed flow/bleeding descriptor
    symptom_severities   : preferred input — {symptom: 0..5}
    model_version        : override MODEL_VERSION from config
    personal_stats       : per-phase personal symptom stats from
                           ``cycle.personalization.compute_personal_stats()``
                           — if provided, L2 probs are softly adjusted toward
                           the user's own symptom baseline before fusion
    context_completeness : fraction of the last 3 days the user actually logged
                           (0.0 = no recent logs, 1.0 = all 3 days logged).
                           Low values boost L1 timing weight in fusion.

    Returns
    -------
    dict with keys: mode, layer1, layer2, layer3,
                    final_phase_probs, final_phase, [model_version]
    """
    symptoms = symptoms or []
    version  = (model_version or MODEL_VERSION).lower()

    layer1              = get_layer1_output(period_starts, today=today)
    layer1_probs_3      = _map_layer1_to_non_menstrual(layer1["phase_probs"])
    layer1_non_men_top  = _get_layer1_non_menstrual_top_phase(layer1_probs_3)

    # ── Override: user explicitly logged a period start today ─────────────────
    if period_start_logged:
        return {
            "mode":   "period_start_override",
            "layer1": layer1,
            "layer2": None,
            "layer3": {
                "timing_status": "Period started",
                "timing_note":   "You logged a period start today, so the cycle resets and today is marked as menstrual.",
                "history_phase": layer1_non_men_top,
                "symptom_phase": None,
            },
            "final_phase_probs": {"Menstrual": 1.0, "Follicular": 0.0, "Fertility": 0.0, "Luteal": 0.0},
            "final_phase":       "Menstrual",
        }

    # ── Auto-detect: ongoing menstrual days based on cycle day + flow ─────────
    if _is_ongoing_menstrual(layer1, flow, recent_daily_logs):
        return {
            "mode":   "menstrual_ongoing",
            "layer1": layer1,
            "layer2": None,
            "layer3": {
                "timing_status": "Menstrual (ongoing)",
                "timing_note":   f"Cycle day {layer1.get('cycle_day')} — within menstrual window.",
                "history_phase": layer1_non_men_top,
                "symptom_phase": None,
            },
            "final_phase_probs": {"Menstrual": 1.0, "Follicular": 0.0, "Fertility": 0.0, "Luteal": 0.0},
            "final_phase":       "Menstrual",
        }

    # ── Fallback: no period history → L1 uniform prior, skip L2 ──────────────
    # cycle_day_norm defaults to 0.5 in the v3/v4 feature builder (mid-cycle),
    # which the model reads as day ~14 and predicts Luteal at ~92% confidence.
    # Return the L1 3-class prior instead of a confidently wrong L2 answer.
    if not period_starts:
        final_phase = max(layer1_probs_3, key=layer1_probs_3.get)
        return {
            "mode":   "layer1_only_no_history",
            "layer1": layer1,
            "layer2": None,
            "layer3": None,
            "final_phase_probs": {
                "Menstrual": 0.0,
                **{p: layer1_probs_3.get(p, 0.0) for p in ["Follicular", "Fertility", "Luteal"]},
            },
            "final_phase": final_phase,
        }

    # ── Fallback: no symptom input → L1 3-class prior, skip L2 ───────────────
    if not has_symptom_input(
        symptoms=symptoms,
        cervical_mucus=cervical_mucus,
        appetite=appetite,
        exerciselevel=exerciselevel,
        recent_daily_logs=recent_daily_logs,
        symptom_severities=symptom_severities,
    ):
        final_phase = max(layer1_probs_3, key=layer1_probs_3.get)
        return {
            "mode":   "layer1_only_non_menstrual",
            "layer1": layer1,
            "layer2": None,
            "layer3": None,
            "final_phase_probs": {
                "Menstrual": 0.0,
                **{p: layer1_probs_3.get(p, 0.0) for p in ["Follicular", "Fertility", "Luteal"]},
            },
            "final_phase": final_phase,
        }

    # ── Full fusion path ──────────────────────────────────────────────────────
    shared_kwargs = dict(
        symptoms=symptoms,
        cervical_mucus=cervical_mucus,
        appetite=appetite,
        exerciselevel=exerciselevel,
        recent_daily_logs=recent_daily_logs,
        flow=flow,
        symptom_severities=symptom_severities,
        layer1=layer1,
        known_starts=period_starts,
    )

    if version == "v5":
        layer2 = get_layer2_v5_output(**shared_kwargs)
    elif version == "v4":
        layer2 = get_layer2_v4_output(**shared_kwargs)
    elif version == "v3":
        layer2 = get_layer2_v3_output(**shared_kwargs)
    else:
        layer2 = get_layer2_v1_output(
            symptoms=symptoms,
            cervical_mucus=cervical_mucus,
            appetite=appetite,
            exerciselevel=exerciselevel,
            recent_daily_logs=recent_daily_logs,
            flow=flow,
        )

    # ── Optional personalisation: nudge L2 probs toward user's own baseline ──
    l2_probs_for_fusion = layer2["phase_probs"]
    personal_summary    = None
    if personal_stats and symptom_severities:
        from .personalization import apply_personal_prior, personal_stats_summary
        l2_probs_before = dict(l2_probs_for_fusion)
        l2_probs_for_fusion = apply_personal_prior(
            l2_probs_before,
            symptom_severities,
            personal_stats,
        )
        personal_summary = personal_stats_summary(personal_stats)

        # ── Flip-logging ──────────────────────────────────────────────────
        # Emit a structured event whenever personalization changes which
        # phase wins (the common case where we care), or — at DEBUG — for
        # every apply_personal_prior call so we can audit the blend.
        top_before = max(l2_probs_before,    key=l2_probs_before.get)
        top_after  = max(l2_probs_for_fusion, key=l2_probs_for_fusion.get)
        flipped    = top_before != top_after
        log_fields = {
            "event":           "personalization_applied",
            "flipped":         flipped,
            "top_before":      top_before,
            "top_after":       top_after,
            "prob_before":     round(float(l2_probs_before[top_before]),    4),
            "prob_after":      round(float(l2_probs_for_fusion[top_after]), 4),
            "blend_weight":    (personal_summary or {}).get("weight"),
            "min_n_real":      min(
                (personal_summary or {}).get("days_by_phase", {"_": 0}).values()
            ),
            "today":           today,
        }
        if flipped:
            _flip_logger.info(
                "personalization_flipped phase %s->%s (w=%s)",
                top_before, top_after, log_fields["blend_weight"],
                extra=log_fields,
            )
        else:
            _flip_logger.debug(
                "personalization_no_flip phase=%s (w=%s)",
                top_after, log_fields["blend_weight"],
                extra=log_fields,
            )

    mucus_score = MUCUS_FERTILITY_MAP.get(_normalize_mucus(cervical_mucus), 0.0)

    fused_probs_3      = _fuse_non_menstrual_probs(
        layer1_probs_3, l2_probs_for_fusion,
        mucus_fertility_score=mucus_score,
        model_version=version,
        context_completeness=context_completeness,
    )
    constrained_probs_3 = _constrain_non_menstrual_probs(
        fused_probs=fused_probs_3,
        baseline_phase=layer1_non_men_top,
        layer2=layer2,
        model_version=version,
        forecast_confidence=layer1.get("forecast_confidence", "high"),
    )

    final_phase = max(constrained_probs_3, key=constrained_probs_3.get)

    layer3 = get_layer3_output(
        layer1={**layer1, "top_phase": layer1_non_men_top, "phase_probs": layer1_probs_3},
        layer2={**layer2, "top_phase": final_phase},
        period_start_logged=False,
    )

    result = {
        "mode":   "fused_non_menstrual",
        "layer1": {**layer1, "non_menstrual_phase_probs": layer1_probs_3, "top_phase": layer1_non_men_top},
        "layer2": layer2,
        "layer3": layer3,
        "final_phase_probs": {
            "Menstrual": 0.0,
            "Follicular": constrained_probs_3.get("Follicular", 0.0),
            "Fertility":  constrained_probs_3.get("Fertility",  0.0),
            "Luteal":     constrained_probs_3.get("Luteal",     0.0),
        },
        "final_phase":   final_phase,
        "model_version": version,
    }
    if personal_summary is not None:
        result["personalization"] = personal_summary
    return result
