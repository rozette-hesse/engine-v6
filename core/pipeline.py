"""
Prediction Pipeline
===================
Orchestrates the full request → response flow for both API routes.
Route handlers delegate all business logic here; they only handle HTTP
concerns (request parsing, response serialisation).

Two entry points
----------------
  run_prediction(payload)  — full pipeline: cycle → phase → recommendations
  run_recommend(payload)   — manual mode: skip cycle, go straight to recommender
"""

from datetime import date as _date, timedelta as _timedelta
from typing import Optional

from cycle import get_fused_output, get_proximity_output, resolve_phase_from_fusion
from recommender import recommend, suggest_phase_mode
from core.transformers import (
    build_cycle_inputs,
    build_recommender_inputs,
    build_symptom_severities,
)


PHASE_LABELS: dict[str, str] = {
    "MEN": "Menstrual",
    "FOL": "Follicular",
    "OVU": "Ovulatory",
    "LUT": "Luteal",
    "LL":  "Late Luteal",
}


# ── Response formatters ───────────────────────────────────────────────────────

def _format_recommendations(raw: dict) -> dict:
    """Flatten recommendation output to the API-stable shape."""
    results = {}
    for domain, categories in raw.items():
        results[domain] = {}
        for category, items in categories.items():
            results[domain][category] = [
                {
                    "id":          item["id"],
                    "text":        item["text"],
                    "final_score": item["final_score"],
                    "confidence":  item["confidence"],
                    "intensity":   item["intensity"],
                    "dur_min":     item["dur_min"],
                    "dur_max":     item["dur_max"],
                    "is_blocked":  item["is_blocked"],
                    "rank":        item.get("rank"),
                }
                for item in items
            ]
    return results


def _extract_fertility(fusion: dict) -> Optional[dict]:
    layer2 = fusion.get("layer2")
    if not layer2:
        return None
    return {
        "status":             layer2.get("fertility_status"),
        "signal_confidence":  layer2.get("signal_confidence"),
        "explanations":       layer2.get("explanations", []),
    }


def _extract_timing(fusion: dict) -> Optional[dict]:
    layer3 = fusion.get("layer3")
    layer1 = fusion.get("layer1") or {}
    if not layer3:
        return None
    return {
        "status":                 layer3.get("timing_status"),
        "note":                   layer3.get("timing_note"),
        "predicted_next_period":  layer1.get("predicted_next_period"),
        "fertile_window":         layer1.get("fertile_window"),
        "ovulation_date":         layer1.get("possible_ovulation_date"),
    }


# ── Pipeline functions ────────────────────────────────────────────────────────

def _proximity_severities(payload) -> dict:
    """Map payload symptom fields to {symptom_name: int 0-5} for the proximity model."""
    return build_symptom_severities(payload)


def signals_from_payload(payload) -> dict:
    """Extract the unified signal dict from any request model."""
    return {
        "ENERGY":            payload.ENERGY.value,
        "STRESS":            payload.STRESS.value,
        "MOOD":              payload.MOOD.value,
        "CRAMP":             payload.CRAMP.value,
        "HEADACHE":          payload.HEADACHE,
        "BLOATING":          payload.BLOATING.value,
        "SLEEP_QUALITY":     payload.SLEEP_QUALITY.value,
        "IRRITABILITY":      payload.IRRITABILITY.value,
        "MOOD_SWING":        payload.MOOD_SWING.value,
        "CRAVINGS":          payload.CRAVINGS.value,
        "FLOW":              payload.FLOW.value,
        "BREAST_TENDERNESS": payload.BREAST_TENDERNESS.value,
        "INDIGESTION":       payload.INDIGESTION.value,
        "BRAIN_FOG":         payload.BRAIN_FOG.value,
    }


def run_prediction(payload) -> dict:
    """
    Full prediction pipeline.

    1. Extract signals from the request payload.
    2. Run the cycle engine (L1 + L2 + L3 fusion).
    3. Resolve the 4-phase cycle output to a 5-phase recommender code.
    4. Score and rank activities via the recommendation engine.
    5. Run the period-proximity classifier.
    6. Assemble and return the unified response dict.
    """
    signals = signals_from_payload(payload)

    # Step 1: build cycle engine inputs (adds CERVICAL_MUCUS, APPETITE, EXERCISE_LEVEL)
    enriched_signals = {
        **signals,
        "CERVICAL_MUCUS":   payload.cervical_mucus.value,
        "APPETITE":         payload.appetite,
        "EXERCISE_LEVEL":   payload.exercise_level,
    }
    cycle_inputs = build_cycle_inputs(enriched_signals)

    # Step 2: fetch history + compute personal stats (skipped when user_id absent)
    log_date           = getattr(payload, "log_date", None) or str(_date.today())
    user_id            = getattr(payload, "user_id", None)
    symptom_severities = build_symptom_severities(payload)
    recent_logs: list  = []
    personal_stats     = {}
    if user_id:
        from db.store import get_recent_logs, get_history_logs, save_log
        from cycle.personalization import compute_personal_stats
        recent_logs    = get_recent_logs(user_id, before_date=log_date)
        history        = get_history_logs(user_id)
        personal_stats = compute_personal_stats(history, reference_date=log_date)

    # Context completeness: fraction of last 3 days the user actually logged.
    # Used by fusion to boost L1 timing weight when symptom history is sparse.
    ref             = _date.fromisoformat(log_date)
    expected_dates  = {str(ref - _timedelta(days=i)) for i in range(1, 4)}
    logged_dates    = {log["date"] for log in recent_logs}
    ctx_completeness = len(expected_dates & logged_dates) / 3.0

    # Step 3: cycle prediction
    fusion = get_fused_output(
        period_starts=payload.period_starts,
        period_start_logged=payload.period_start_logged,
        today=None,
        recent_daily_logs=recent_logs or None,
        symptom_severities=symptom_severities,
        personal_stats=personal_stats or None,
        context_completeness=ctx_completeness,
        **{k: v for k, v in cycle_inputs.items() if k != "symptoms"},
    )

    # Persist full inputs + outputs (after prediction so we can store the result)
    if user_id:
        save_log(
            user_id,
            log_date,
            cycle_inputs,
            period_starts=payload.period_starts,
            period_start_logged=payload.period_start_logged,
            symptom_severities=symptom_severities,
            fusion_output=fusion,
        )

    # Step 4: phase resolution (4-phase → 5-phase)
    phase_code = resolve_phase_from_fusion(fusion)

    # Step 5: recommendations
    # Pick phase_aware vs phase_blind based on how much we trust the cycle
    # output. Low-confidence timing or no-history fallbacks → blind, so
    # symptom-driven rules still produce good suggestions without being
    # diluted by mis-applied periodization guidance.
    phase_mode    = suggest_phase_mode(fusion)
    engine_inputs = build_recommender_inputs(signals, phase_code)
    raw           = recommend(engine_inputs, top_n=payload.top_n, phase_mode=phase_mode)

    # Step 6: period proximity (symptom-only, no timing)
    proximity = get_proximity_output(
        symptoms=cycle_inputs.get("symptoms", []),
        appetite=payload.appetite,
        exerciselevel=payload.exercise_level,
        symptom_severities=_proximity_severities(payload),
    )

    return {
        "phase": {
            "code":         phase_code,
            "label":        PHASE_LABELS.get(phase_code, phase_code),
            "confidence":   fusion.get("layer1", {}).get("forecast_confidence"),
            "regularity":   fusion.get("layer1", {}).get("regularity_status"),
            "mode":         fusion.get("mode"),
            "cycle_day":    fusion.get("layer1", {}).get("cycle_day"),
            "cycle_length": fusion.get("layer1", {}).get("estimated_cycle_length"),
            "phase_probs":  fusion.get("final_phase_probs"),
        },
        "fertility":         _extract_fertility(fusion),
        "timing":            _extract_timing(fusion),
        "period_proximity":  proximity,
        "recommendations":   _format_recommendations(raw),
        "recommender_mode":  phase_mode,
        "personalization":   fusion.get("personalization"),
        "context_completeness": round(ctx_completeness, 2),
    }


def run_recommend(payload) -> dict:
    """
    Manual recommendation pipeline.

    User selects their phase directly — cycle prediction is skipped.
    """
    signals       = signals_from_payload(payload)
    engine_inputs = build_recommender_inputs(signals, payload.PHASE.value)
    raw           = recommend(engine_inputs, top_n=payload.top_n)
    return {"results": _format_recommendations(raw)}
