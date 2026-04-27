"""
Personalization Layer
=====================
Adjusts L2 phase probabilities using a user's own symptom history.

Four mechanisms address the main real-world user behaviour problems:

1. Bayesian warm start
   Personal means are initialised from population priors (POPULATION_PRIORS)
   weighted as PRIOR_STRENGTH virtual days.  New users get sensible defaults
   from day one; data gradually shifts the means toward the individual.
   Cold start is solved: personalization is always active, just weak at first.

2. Recency decay
   Each historical log is weighted by exp(-age / DECAY_HALFLIFE).  Logs from
   90 days ago count half as much; logs from 6 months ago count ~15%.
   Handles abandonment-and-return: old data from a different health state
   (pregnancy, illness, medication) fades without manual deletion.

3. Cosine similarity likelihood (relative normalization)
   Instead of comparing absolute symptom levels, the model computes the
   cosine similarity between today's symptom vector and each phase's mean
   vector.  This is scale-invariant: the signal is the *pattern* of which
   symptoms are elevated, not their absolute magnitude.
   Handles symptomatic-logging bias: a user who only logs when feeling bad
   has uniformly high phase means, but the per-phase symptom *pattern* still
   discriminates — Luteal elevates cramps+mood; Fertility elevates
   breast-tenderness; Follicular is low across the board.

4. Context completeness (handled in fusion.py, computed in pipeline.py)
   When recent context logs are missing (logging gap), fusion.py boosts
   the L1 timing weight so the calendar carries more weight than absent
   symptom history.  See _fuse_non_menstrual_probs in fusion.py.

Population priors
-----------------
POPULATION_PRIORS contains clinically-informed per-phase symptom means on
the 0-5 severity scale, derived from published menstrual symptom literature.
Replace or supplement with data-derived means by running:
  python scripts/compute_population_priors.py
and saving the result to artifacts/population_phase_means.json — the
load_population_priors() helper will pick it up automatically.

Tuning constants
----------------
  PRIOR_STRENGTH      = 10    virtual days of population data
  DECAY_HALFLIFE      = 90    days (half-weight at 90 days)
  MAX_PERSONAL_WEIGHT = 0.35  ceiling on personal-history influence in blend
"""

import json
import math
import os
from datetime import date as _date
from pathlib import Path
from typing import Dict, List, Optional

from .config import ARTIFACTS_DIR, NON_MENSTRUAL_PHASES, SUPPORTED_SYMPTOMS

# ── Tuning ────────────────────────────────────────────────────────────────────

PRIOR_STRENGTH      = 10    # virtual days — prior dominates until this many real logs
DECAY_HALFLIFE      = 90    # days; log weight = exp(-age / DECAY_HALFLIFE)
MAX_PERSONAL_WEIGHT = float(os.environ.get("PERSONAL_WEIGHT_MAX", 0.35))
RAMP_HALFPOINT      = float(os.environ.get("PERSONAL_WEIGHT_RAMP", 30.0))
# Weight at min_n_real n : MAX_PERSONAL_WEIGHT × n / (n + RAMP_HALFPOINT)
# Env vars let scripts/evaluate_personalization.py sweep values without code edits.

# Hard safety cap — no config can push the personalization blend above this.
# Derived from the weight×history grid eval: above ~0.5 the cosine-similarity
# likelihood starts drowning out the trained L2 classifier and accuracy
# collapses (up to −12pp overall at weight≈0.8). See scripts/evaluate_personalization.py
# --history-sweep for the evidence.
PERSONAL_WEIGHT_HARD_CAP = 0.50

# Discriminability gate — if a user's own phase symptom-means are all near-
# parallel (no phase-specific signature), cosine similarity cannot discriminate
# and personalization becomes noise. Skip the blend entirely when the MAX
# pairwise cosine similarity between phase means exceeds this threshold.
DISCRIMINABILITY_MAX_COSINE = 0.97


# ── Population priors ─────────────────────────────────────────────────────────
# Clinically-informed per-phase mean symptom severity (0-5 scale).
# Follicular: post-menstrual recovery — low overall symptom burden.
# Fertility:  peak estrogen — low fatigue, notable breast tenderness.
# Luteal:     PMS window — elevated pain, fatigue, mood, bloating.

_HARDCODED_PRIORS: Dict[str, Dict[str, float]] = {
    "Follicular": {
        "headaches":    0.6,
        "cramps":       0.3,
        "sorebreasts":  0.5,
        "fatigue":      0.8,
        "sleepissue":   0.5,
        "moodswing":    0.6,
        "stress":       1.0,
        "foodcravings": 0.6,
        "indigestion":  0.4,
        "bloating":     0.5,
    },
    "Fertility": {
        "headaches":    0.8,
        "cramps":       0.7,
        "sorebreasts":  1.2,
        "fatigue":      0.6,
        "sleepissue":   0.4,
        "moodswing":    0.5,
        "stress":       0.9,
        "foodcravings": 0.7,
        "indigestion":  0.5,
        "bloating":     0.7,
    },
    "Luteal": {
        "headaches":    1.3,
        "cramps":       1.0,
        "sorebreasts":  1.5,
        "fatigue":      1.8,
        "sleepissue":   1.3,
        "moodswing":    1.8,
        "stress":       1.6,
        "foodcravings": 1.5,
        "indigestion":  1.0,
        "bloating":     1.5,
    },
}


def load_population_priors() -> Dict[str, Dict[str, float]]:
    """
    Load population priors from artifacts/ if available, else use hardcoded values.

    The artifact file (population_phase_means.json) is generated by running:
      python scripts/compute_population_priors.py

    Expected format::

        {
            "Follicular": {"headaches": 0.62, "cramps": 0.28, ...},
            "Fertility":  {...},
            "Luteal":     {...}
        }
    """
    artifact_path = ARTIFACTS_DIR / "population_phase_means.json"
    if artifact_path.exists():
        try:
            with open(artifact_path) as f:
                loaded = json.load(f)
            # Validate all phases and symptoms present
            if all(
                phase in loaded and all(s in loaded[phase] for s in SUPPORTED_SYMPTOMS)
                for phase in NON_MENSTRUAL_PHASES
            ):
                return loaded
        except Exception:
            pass
    return _HARDCODED_PRIORS


POPULATION_PRIORS: Dict[str, Dict[str, float]] = load_population_priors()


# ── Cosine similarity ─────────────────────────────────────────────────────────

def _cosine_likelihood(
    today_vec: Dict[str, float],
    phase_mean: Dict[str, float],
) -> float:
    """
    Cosine similarity between today's symptom vector and a phase mean, mapped
    to [0, 1].

    Scale-invariant: a user who logs uniformly high symptoms will still have
    phase likelihoods driven by the *pattern* of which symptoms are elevated,
    not their absolute magnitude.  This handles symptomatic-logging bias.

    Returns 0.5 (neutral) when either vector is near-zero — no information
    to discriminate on, so all phases are treated as equally likely.
    """
    dot    = sum(today_vec[s] * phase_mean[s] for s in SUPPORTED_SYMPTOMS)
    norm_t = math.sqrt(sum(today_vec[s] ** 2 for s in SUPPORTED_SYMPTOMS))
    norm_p = math.sqrt(sum(phase_mean[s] ** 2 for s in SUPPORTED_SYMPTOMS))

    if norm_t < 1e-6 or norm_p < 1e-6:
        return 0.5  # neutral: can't compute meaningful similarity

    cos = max(-1.0, min(1.0, dot / (norm_t * norm_p)))
    return (1.0 + cos) / 2.0  # map [-1, 1] → [0, 1]


def _raw_cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Plain cosine similarity in [-1, 1] between two symptom vectors. Returns
    1.0 when either vector is zero — safe "these vectors can't be distinguished"."""
    dot    = sum(a[s] * b[s] for s in SUPPORTED_SYMPTOMS)
    norm_a = math.sqrt(sum(a[s] ** 2 for s in SUPPORTED_SYMPTOMS))
    norm_b = math.sqrt(sum(b[s] ** 2 for s in SUPPORTED_SYMPTOMS))
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 1.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def _max_pairwise_phase_cosine(
    personal_stats: Dict[str, Dict],
    phases: List[str],
) -> float:
    """
    Maximum pairwise cosine similarity between a user's per-phase symptom-mean
    vectors. High value → phase means point in (nearly) the same direction →
    user's phases are not symptom-distinguishable for this user.
    Used to gate personalization: if the user's own phase patterns don't
    discriminate, the cosine-similarity blend adds noise rather than signal.
    """
    if len(phases) < 2:
        return 1.0  # can't discriminate with fewer than 2 phases
    means = [personal_stats[p]["mean"] for p in phases]
    worst = 0.0
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            c = _raw_cosine(means[i], means[j])
            if c > worst:
                worst = c
    return worst


# ── Phase stats ───────────────────────────────────────────────────────────────

def compute_personal_stats(
    history: List[Dict],
    reference_date: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Compute Bayesian per-phase symptom means from a user's stored history.

    Combines four ideas:

    - Bayesian warm start: every phase always has a mean (population prior +
      user data), so personalization is never completely silent.
    - Recency decay: older logs contribute exponentially less.
    - Weighted means: each log's contribution is decay_weight × severity.
    - Grand mean: overall mean across all phases, used by cosine similarity
      to understand the user's general symptom baseline.

    Parameters
    ----------
    history        : list of dicts from ``db.store.get_history_logs()``
                     Each dict must have "date", "final_phase",
                     and "symptom_severities" keys.
    reference_date : YYYY-MM-DD string for computing log ages.
                     Defaults to today.

    Returns
    -------
    Dict with keys:

    - "Follicular", "Fertility", "Luteal" → each:
        {
            "mean":         {symptom: float},   # Bayesian mean (prior + real)
            "n_real":       int,                # actual days logged by user
            "total_weight": float,              # decay-weighted sum of days
        }
    - "_grand_mean" → {"mean": {symptom: float}, "total_weight": float}
      Overall symptom baseline across all phases (for reference / debugging).
    """
    ref = _date.fromisoformat(reference_date) if reference_date else _date.today()

    # Accumulate decay-weighted symptom sums per phase
    phase_acc: Dict[str, Dict] = {
        phase: {
            "weighted_sum":  {s: 0.0 for s in SUPPORTED_SYMPTOMS},
            "total_weight":  0.0,
            "n_real":        0,
        }
        for phase in NON_MENSTRUAL_PHASES
    }

    # Accumulate across ALL logged days (for grand mean)
    grand_sum    = {s: 0.0 for s in SUPPORTED_SYMPTOMS}
    grand_weight = 0.0

    for day in history:
        # ── Recency decay ──────────────────────────────────────────────────
        try:
            log_d    = _date.fromisoformat(day["date"])
            age_days = max(0, (ref - log_d).days)
        except (KeyError, ValueError, TypeError):
            age_days = 0
        decay_w = math.exp(-age_days / DECAY_HALFLIFE)

        sev = day.get("symptom_severities") or {}

        # Grand mean: all logged days regardless of phase
        for sym in SUPPORTED_SYMPTOMS:
            grand_sum[sym] += decay_w * float(sev.get(sym, 0) or 0)
        grand_weight += decay_w

        # Per-phase accumulation (non-menstrual only)
        phase = day.get("final_phase")
        if phase not in NON_MENSTRUAL_PHASES:
            continue

        acc = phase_acc[phase]
        for sym in SUPPORTED_SYMPTOMS:
            acc["weighted_sum"][sym] += decay_w * float(sev.get(sym, 0) or 0)
        acc["total_weight"] += decay_w
        acc["n_real"]       += 1

    # ── Bayesian per-phase means ──────────────────────────────────────────────
    # mean = (PRIOR_STRENGTH × prior_mean + Σ(decay_w × severity))
    #        / (PRIOR_STRENGTH + Σ decay_w)
    # As n_real → 0 : mean → population prior
    # As n_real → ∞ : mean → user's own weighted mean
    stats: Dict[str, Dict] = {}
    for phase in NON_MENSTRUAL_PHASES:
        acc   = phase_acc[phase]
        prior = POPULATION_PRIORS[phase]
        w     = acc["total_weight"]
        denom = PRIOR_STRENGTH + w

        stats[phase] = {
            "mean": {
                sym: (PRIOR_STRENGTH * prior[sym] + acc["weighted_sum"][sym]) / denom
                for sym in SUPPORTED_SYMPTOMS
            },
            "n_real":       acc["n_real"],
            "total_weight": w,
        }

    # ── Grand mean ────────────────────────────────────────────────────────────
    if grand_weight > 0:
        grand_mean = {s: grand_sum[s] / grand_weight for s in SUPPORTED_SYMPTOMS}
    else:
        grand_mean = {s: 0.0 for s in SUPPORTED_SYMPTOMS}

    stats["_grand_mean"] = {
        "mean":         grand_mean,
        "total_weight": grand_weight,
    }

    return stats


# ── Probability adjustment ────────────────────────────────────────────────────

def apply_personal_prior(
    l2_probs: Dict[str, float],
    today_severities: Dict[str, int],
    personal_stats: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Blend L2 probabilities with a personal cosine-similarity likelihood.

    For each phase, evaluates the cosine similarity between today's symptom
    pattern and this user's typical symptom pattern for that phase (Bayesian
    mean).  Normalises similarities to a probability distribution and blends
    with L2 at a weight that grows with real logged data.

    Because Bayesian warm start ensures all three phases always have a mean,
    there is no partial-coverage logic needed — the blend weight smoothly
    starts near 0 (no real data) and ramps toward MAX_PERSONAL_WEIGHT.

    Parameters
    ----------
    l2_probs         : {phase: probability} from the L2 model
    today_severities : {symptom: 0-5} for today
    personal_stats   : output of ``compute_personal_stats()``

    Returns
    -------
    Adjusted {phase: probability} normalised to sum to 1.
    Returns l2_probs unchanged if personal_stats is empty or malformed.
    """
    if not personal_stats:
        return l2_probs

    phases = [p for p in NON_MENSTRUAL_PHASES if p in personal_stats]
    if not phases:
        return l2_probs

    today_vec = {s: float(today_severities.get(s, 0) or 0) for s in SUPPORTED_SYMPTOMS}

    # ── Discriminability gate ─────────────────────────────────────────────────
    # If a user's per-phase means are near-parallel vectors, cosine similarity
    # cannot discriminate between them — personalization becomes noise and can
    # drown out the trained L2 classifier. Skip the blend.
    if _max_pairwise_phase_cosine(personal_stats, phases) >= DISCRIMINABILITY_MAX_COSINE:
        return l2_probs

    # ── Cosine similarity likelihood ──────────────────────────────────────────
    similarities = {
        phase: _cosine_likelihood(today_vec, personal_stats[phase]["mean"])
        for phase in phases
    }

    total_sim = sum(similarities.values())
    if total_sim <= 0:
        return l2_probs

    personal_probs = {p: v / total_sim for p, v in similarities.items()}

    # ── Blend weight ──────────────────────────────────────────────────────────
    # Uses the phase with the fewest real logs as the bottleneck.
    # Defaults ramp asymptotically: 0 at n=0, ~17% at n=30, ~26% at n=60, max 35%.
    # Hard cap prevents any config from pushing into the regression zone.
    min_n_real = min(personal_stats[p]["n_real"] for p in phases)
    weight     = MAX_PERSONAL_WEIGHT * min_n_real / (min_n_real + RAMP_HALFPOINT)
    weight     = min(weight, PERSONAL_WEIGHT_HARD_CAP)

    # ── Weighted blend ────────────────────────────────────────────────────────
    blended = {
        phase: (1.0 - weight) * l2_probs.get(phase, 0.0)
               + weight       * personal_probs[phase]
        for phase in phases
    }

    total_blended = sum(blended.values())
    if total_blended <= 0:
        return l2_probs

    return {p: v / total_blended for p, v in blended.items()}


# ── API summary ───────────────────────────────────────────────────────────────

def personal_stats_summary(personal_stats: Dict[str, Dict]) -> Optional[Dict]:
    """
    Return a lightweight summary for the API response.

    Reports whether personalisation is meaningfully active (real data logged),
    the blend weight applied, data coverage per phase, and — while data is
    still accumulating — which phases need more logs.

    A phase is considered "personalised" when n_real >= PRIOR_STRENGTH (10):
    the user's own data has equal or greater weight than the population prior.
    """
    if not personal_stats:
        return None

    phases = [p for p in NON_MENSTRUAL_PHASES if p in personal_stats]
    if not phases:
        return None

    min_n_real = min(personal_stats[p]["n_real"] for p in phases)
    weight     = MAX_PERSONAL_WEIGHT * min_n_real / (min_n_real + RAMP_HALFPOINT)

    days_by_phase      = {p: personal_stats[p]["n_real"] for p in phases}
    personalised_phases = [p for p in phases if personal_stats[p]["n_real"] >= PRIOR_STRENGTH]
    accumulating_phases = [p for p in phases if personal_stats[p]["n_real"] <  PRIOR_STRENGTH]

    return {
        "active":               min_n_real > 0,
        "weight":               round(weight, 3),
        "days_by_phase":        days_by_phase,
        "personalised_phases":  personalised_phases,   # data > population prior
        "accumulating_phases":  accumulating_phases,   # still prior-dominated
    }
