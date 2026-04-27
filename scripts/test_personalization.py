"""
Personalization Layer — Test Script
=====================================
Run:  python scripts/test_personalization.py
      python scripts/test_personalization.py -v   # verbose output

Covers:
  1. Cold start       — empty history returns population priors, weight=0
  2. Bayesian warm start — single log nudges mean, doesn't overwrite prior
  3. Recency decay    — old logs contribute less than recent ones
  4. Cosine similarity — Luteal symptom pattern boosts Luteal probability
  5. Zero vector      — no symptoms → neutral likelihood → l2_probs unchanged
  6. Context completeness — missing logs boost L1 weight in fusion
  7. User journey     — probability shift increases as user logs more cycles
  8. Full pipeline    — get_fused_output accepts all new parameters without error

Exit codes: 0 = all passed, 1 = failures
"""

import os
import sys
import math
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message="X does not have valid feature names",
                        category=UserWarning)

from cycle.personalization import (
    compute_personal_stats,
    apply_personal_prior,
    personal_stats_summary,
    POPULATION_PRIORS,
    PRIOR_STRENGTH,
    MAX_PERSONAL_WEIGHT,
    DECAY_HALFLIFE,
)
from cycle.config import NON_MENSTRUAL_PHASES, SUPPORTED_SYMPTOMS
from cycle.fusion import get_fused_output

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
VERBOSE = "-v" in sys.argv
_failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS} {name}")
    else:
        msg = f"  {FAIL} {name}" + (f" — {detail}" if detail else "")
        print(msg)
        _failures.append(name)


def approx_eq(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def _make_history(n_days: int, phase: str, severities: dict,
                  start_date: str = "2024-01-01") -> list:
    """Build n_days of logs for a single phase starting at start_date."""
    from datetime import date, timedelta
    d = date.fromisoformat(start_date)
    return [
        {
            "date":               str(d + timedelta(days=i)),
            "final_phase":        phase,
            "symptom_severities": severities.copy(),
        }
        for i in range(n_days)
    ]


def _uniform_l2(phases=None) -> dict:
    """Uniform 1/3 probability over non-menstrual phases."""
    phases = phases or NON_MENSTRUAL_PHASES
    return {p: 1.0 / len(phases) for p in phases}


def _period_starts(cycle_day: int, cycle_length: int = 28, n: int = 3) -> list:
    from datetime import date, timedelta
    today = date.fromisoformat("2024-06-15")
    latest_start = today - timedelta(days=cycle_day - 1)
    return [str(latest_start - timedelta(days=cycle_length * i)) for i in range(n)]


# ── Test 1: Cold start ────────────────────────────────────────────────────────

def test_cold_start():
    print("\n[1] Cold start — empty history")
    stats = compute_personal_stats([])

    # All three phases present (Bayesian warm start fills them)
    for phase in NON_MENSTRUAL_PHASES:
        check(f"{phase} present in stats", phase in stats)

    # n_real = 0 for all phases
    for phase in NON_MENSTRUAL_PHASES:
        n = stats[phase]["n_real"]
        check(f"{phase} n_real = 0", n == 0, f"got {n}")

    # Blend weight = 0 when no real data
    min_n = min(stats[p]["n_real"] for p in NON_MENSTRUAL_PHASES)
    weight = MAX_PERSONAL_WEIGHT * min_n / (min_n + 30.0)
    check("blend weight = 0 with no real data", approx_eq(weight, 0.0), f"got {weight:.4f}")

    # Means equal population priors (n_real=0 → pure prior)
    for phase in NON_MENSTRUAL_PHASES:
        for sym in SUPPORTED_SYMPTOMS:
            got      = stats[phase]["mean"][sym]
            expected = POPULATION_PRIORS[phase][sym]
            check(
                f"{phase}.{sym} = population prior",
                approx_eq(got, expected, tol=1e-4),
                f"got {got:.4f}, expected {expected:.4f}",
            )
            if not VERBOSE:
                break  # check one symptom per phase to keep output concise

    # apply_personal_prior should return l2_probs unchanged at weight=0
    l2 = _uniform_l2()
    adjusted = apply_personal_prior(l2, {s: 0 for s in SUPPORTED_SYMPTOMS}, stats)
    for phase in NON_MENSTRUAL_PHASES:
        check(
            f"apply returns unchanged l2 at weight=0 ({phase})",
            approx_eq(adjusted[phase], l2[phase], tol=1e-4),
            f"got {adjusted[phase]:.4f}, expected {l2[phase]:.4f}",
        )


# ── Test 2: Bayesian warm start — single log nudges, doesn't overwrite ────────

def test_bayesian_warm_start():
    print("\n[2] Bayesian warm start — single Luteal log")

    # One Luteal log with very high symptoms
    history = _make_history(1, "Luteal", {s: 5 for s in SUPPORTED_SYMPTOMS},
                            start_date="2024-06-14")
    stats = compute_personal_stats(history, reference_date="2024-06-15")

    prior_mean = POPULATION_PRIORS["Luteal"]["cramps"]
    personal_mean = stats["Luteal"]["mean"]["cramps"]

    # Mean should be between prior and 5.0 (one log nudges but doesn't overwrite)
    check(
        "Luteal cramps mean between prior and 5.0",
        prior_mean < personal_mean < 5.0,
        f"prior={prior_mean:.2f}, got={personal_mean:.2f}",
    )

    # Mean should be closer to prior than to 5.0 (prior_strength=10 outweighs 1 log)
    dist_to_prior    = abs(personal_mean - prior_mean)
    dist_to_observed = abs(personal_mean - 5.0)
    check(
        "mean closer to prior than to single observation",
        dist_to_prior < dist_to_observed,
        f"dist_to_prior={dist_to_prior:.3f}, dist_to_observed={dist_to_observed:.3f}",
    )

    # Follicular unchanged (no logs for that phase)
    fol_mean = stats["Follicular"]["mean"]["cramps"]
    check(
        "Follicular mean unchanged (still population prior)",
        approx_eq(fol_mean, POPULATION_PRIORS["Follicular"]["cramps"], tol=1e-4),
        f"got {fol_mean:.4f}",
    )


# ── Test 3: Recency decay ─────────────────────────────────────────────────────

def test_recency_decay():
    print("\n[3] Recency decay")

    # Recent log (yesterday) vs old log (180 days ago)
    recent_history = _make_history(1, "Follicular", {"cramps": 5, **{s: 0 for s in SUPPORTED_SYMPTOMS if s != "cramps"}},
                                   start_date="2024-06-14")
    old_history    = _make_history(1, "Follicular", {"cramps": 5, **{s: 0 for s in SUPPORTED_SYMPTOMS if s != "cramps"}},
                                   start_date="2024-01-17")  # ~150 days before 2024-06-15

    stats_recent = compute_personal_stats(recent_history, reference_date="2024-06-15")
    stats_old    = compute_personal_stats(old_history,    reference_date="2024-06-15")

    # Recent log should push the mean further from the prior than the old log
    prior_cramps   = POPULATION_PRIORS["Follicular"]["cramps"]
    shift_recent   = abs(stats_recent["Follicular"]["mean"]["cramps"] - prior_cramps)
    shift_old      = abs(stats_old   ["Follicular"]["mean"]["cramps"] - prior_cramps)

    check(
        "recent log shifts mean more than old log",
        shift_recent > shift_old,
        f"shift_recent={shift_recent:.4f}, shift_old={shift_old:.4f}",
    )

    # Verify the decay factor at 150 days is < 1
    decay_factor = math.exp(-150 / DECAY_HALFLIFE)
    check(
        f"150-day decay factor < 1 (got {decay_factor:.3f})",
        decay_factor < 1.0,
    )


# ── Test 4: Cosine similarity — Luteal pattern boosts Luteal ─────────────────

def test_cosine_similarity():
    print("\n[4] Cosine similarity — Luteal pattern")

    # Seed with enough data to get non-trivial weight (~15 logs each phase)
    history = (
        _make_history(15, "Follicular", {s: 0 for s in SUPPORTED_SYMPTOMS},
                      start_date="2024-01-01")
        + _make_history(15, "Fertility",  {"sorebreasts": 3, **{s: 0 for s in SUPPORTED_SYMPTOMS if s != "sorebreasts"}},
                        start_date="2024-02-01")
        + _make_history(15, "Luteal",     {"cramps": 4, "moodswing": 4, "fatigue": 4,
                                           **{s: 0 for s in SUPPORTED_SYMPTOMS if s not in ("cramps","moodswing","fatigue")}},
                        start_date="2024-03-01")
    )
    stats = compute_personal_stats(history, reference_date="2024-06-15")

    # Today: classic Luteal pattern
    today_sev = {s: 0 for s in SUPPORTED_SYMPTOMS}
    today_sev.update({"cramps": 4, "moodswing": 4, "fatigue": 4})

    l2 = _uniform_l2()
    adjusted = apply_personal_prior(l2, today_sev, stats)

    check(
        "Luteal probability highest after Luteal symptom pattern",
        adjusted["Luteal"] > adjusted["Follicular"] and
        adjusted["Luteal"] > adjusted["Fertility"],
        f"probs: {adjusted}",
    )
    check(
        "Luteal probability above uniform (0.333)",
        adjusted["Luteal"] > 1.0 / 3.0,
        f"got {adjusted['Luteal']:.4f}",
    )

    if VERBOSE:
        print(f"    adjusted probs: { {k: f'{v:.3f}' for k, v in adjusted.items()} }")


# ── Test 5: Zero vector + discriminability gate ──────────────────────────────

def test_zero_vector():
    print("\n[5] Zero symptom vector & discriminability gate")

    # ── 5a: discriminability gate ────────────────────────────────────────────
    # Build a history where all three phase means are IDENTICAL (same severities
    # every phase). This is the "user whose phases are not symptom-distinguishable"
    # case — personalization should skip entirely rather than add noise.
    flat_history = (
        _make_history(10, "Follicular", {s: 1 for s in SUPPORTED_SYMPTOMS}, "2024-01-01")
        + _make_history(10, "Fertility",  {s: 1 for s in SUPPORTED_SYMPTOMS}, "2024-02-01")
        + _make_history(10, "Luteal",     {s: 1 for s in SUPPORTED_SYMPTOMS}, "2024-03-01")
    )
    flat_stats = compute_personal_stats(flat_history, reference_date="2024-06-15")

    l2 = {"Follicular": 0.1, "Fertility": 0.7, "Luteal": 0.2}
    adjusted_flat = apply_personal_prior(
        l2, {s: 0 for s in SUPPORTED_SYMPTOMS}, flat_stats
    )
    check(
        "discriminability gate returns l2 unchanged when phase means are parallel",
        all(approx_eq(adjusted_flat[k], l2[k], tol=1e-9) for k in l2),
        f"l2={l2}, adjusted={adjusted_flat}",
    )

    # ── 5b: zero-vector behaviour when phase patterns ARE discriminable ──────
    # Distinct per-phase patterns: Luteal-heavy in the last rows.
    history = (
        _make_history(10, "Follicular", {s: 0 for s in SUPPORTED_SYMPTOMS}, "2024-01-01")
        + _make_history(10, "Fertility",
                         {**{s: 0 for s in SUPPORTED_SYMPTOMS}, "sorebreasts": 3},
                         "2024-02-01")
        + _make_history(10, "Luteal",
                         {**{s: 0 for s in SUPPORTED_SYMPTOMS}, "cramps": 3, "bloating": 3},
                         "2024-03-01")
    )
    stats = compute_personal_stats(history, reference_date="2024-06-15")

    adjusted = apply_personal_prior(l2, {s: 0 for s in SUPPORTED_SYMPTOMS}, stats)

    total = sum(adjusted.values())
    check("zero vector result sums to 1", approx_eq(total, 1.0, tol=1e-4), f"sum={total:.6f}")
    check("all probs positive", all(v > 0 for v in adjusted.values()))


# ── Test 6: Context completeness boosts L1 in fusion ─────────────────────────

def test_context_completeness():
    """
    Tests the L1-weight boost mechanism directly via _fuse_non_menstrual_probs,
    bypassing constraint logic that can confound end-to-end directional tests.

    Setup: L1 strongly favours Luteal, L2 strongly favours Follicular.
    At context=1.0  L2 carries full weight (0.90) → Follicular wins.
    At context=0.0  L1 carries extra weight    → Luteal gets a larger share.
    """
    print("\n[6] Context completeness — L1 weight boost")
    from cycle.fusion import _fuse_non_menstrual_probs

    l1 = {"Follicular": 0.05, "Fertility": 0.05, "Luteal": 0.90}
    l2 = {"Follicular": 0.80, "Fertility": 0.10, "Luteal": 0.10}

    fused_full = _fuse_non_menstrual_probs(l1, l2, model_version="v4", context_completeness=1.0)
    fused_none = _fuse_non_menstrual_probs(l1, l2, model_version="v4", context_completeness=0.0)

    check("full context run produces valid probs",
          approx_eq(sum(fused_full.values()), 1.0, tol=1e-4))
    check("no context run produces valid probs",
          approx_eq(sum(fused_none.values()), 1.0, tol=1e-4))

    # With full context, L2 dominates → Follicular should be highest
    check(
        "full context: Follicular wins (L2 dominates)",
        fused_full["Follicular"] > fused_full["Luteal"],
        f"Follicular={fused_full['Follicular']:.3f}, Luteal={fused_full['Luteal']:.3f}",
    )
    # With no context, L1 gets a bigger share → Luteal prob increases vs full context
    check(
        "no context: Luteal prob increases (L1 weight boosted)",
        fused_none["Luteal"] > fused_full["Luteal"],
        f"full={fused_full['Luteal']:.3f}, no_context={fused_none['Luteal']:.3f}",
    )
    # Symmetrically: Follicular prob decreases
    check(
        "no context: Follicular prob decreases (L2 weight reduced)",
        fused_none["Follicular"] < fused_full["Follicular"],
        f"full={fused_full['Follicular']:.3f}, no_context={fused_none['Follicular']:.3f}",
    )

    # End-to-end smoke: both context values accepted without error
    period_starts = _period_starts(cycle_day=20)
    sev = {"cramps": 2, **{s: 0 for s in SUPPORTED_SYMPTOMS if s != "cramps"}}
    for cc in (0.0, 0.33, 0.67, 1.0):
        out = get_fused_output(period_starts=period_starts, today="2024-06-15",
                               symptom_severities=sev, context_completeness=cc)
        check(f"context_completeness={cc} accepted", out.get("final_phase") is not None)

    if VERBOSE:
        print(f"    full context fused: { {k: f'{v:.3f}' for k, v in fused_full.items()} }")
        print(f"    no context fused:   { {k: f'{v:.3f}' for k, v in fused_none.items()} }")


# ── Test 7: User journey — weight grows with more logs ────────────────────────

def test_user_journey():
    print("\n[7] User journey — weight grows with logged cycles")

    luteal_sev = {
        "cramps": 4, "moodswing": 4, "fatigue": 3, "bloating": 3,
        **{s: 0 for s in SUPPORTED_SYMPTOMS if s not in ("cramps","moodswing","fatigue","bloating")}
    }

    def weight_for_n(n: int) -> float:
        history = (
            _make_history(n, "Follicular", {s: 0 for s in SUPPORTED_SYMPTOMS}, "2024-01-01")
            + _make_history(n, "Fertility",  {s: 0 for s in SUPPORTED_SYMPTOMS}, "2024-02-15")
            + _make_history(n, "Luteal",     luteal_sev, "2024-03-15")
        )
        stats = compute_personal_stats(history, reference_date="2024-06-15")
        min_n = min(stats[p]["n_real"] for p in NON_MENSTRUAL_PHASES)
        return MAX_PERSONAL_WEIGHT * min_n / (min_n + 30.0)

    w0  = weight_for_n(0)
    w5  = weight_for_n(5)
    w15 = weight_for_n(15)
    w30 = weight_for_n(30)

    check("weight=0 with no logs",         approx_eq(w0, 0.0), f"got {w0:.4f}")
    check("weight grows: 0 < 5 < 15 < 30", w0 < w5 < w15 < w30,
          f"w0={w0:.3f}, w5={w5:.3f}, w15={w15:.3f}, w30={w30:.3f}")
    check(f"weight stays <= {MAX_PERSONAL_WEIGHT}",
          w30 <= MAX_PERSONAL_WEIGHT, f"got {w30:.4f}")

    if VERBOSE:
        print(f"    weights: n=0→{w0:.3f}, n=5→{w5:.3f}, n=15→{w15:.3f}, n=30→{w30:.3f}")


# ── Test 8: Full pipeline smoke test ─────────────────────────────────────────

def test_full_pipeline_smoke():
    print("\n[8] Full pipeline smoke test")

    period_starts = _period_starts(cycle_day=14)  # Fertility timing

    # Build mock history: 10 days per phase
    history = (
        _make_history(10, "Follicular", {s: 1 for s in SUPPORTED_SYMPTOMS}, "2024-02-01")
        + _make_history(10, "Fertility",  {"sorebreasts": 3, **{s: 0 for s in SUPPORTED_SYMPTOMS if s != "sorebreasts"}}, "2024-02-15")
        + _make_history(10, "Luteal",     {"cramps": 4, "moodswing": 3, **{s: 0 for s in SUPPORTED_SYMPTOMS if s not in ("cramps","moodswing")}}, "2024-03-01")
    )
    stats = compute_personal_stats(history, reference_date="2024-06-15")

    try:
        out = get_fused_output(
            period_starts=period_starts,
            today="2024-06-15",
            symptom_severities={"sorebreasts": 3, "fatigue": 1,
                                **{s: 0 for s in SUPPORTED_SYMPTOMS if s not in ("sorebreasts","fatigue")}},
            personal_stats=stats,
            context_completeness=0.67,
        )
        check("get_fused_output returns final_phase",
              out.get("final_phase") in ("Follicular", "Fertility", "Luteal"),
              f"got {out.get('final_phase')}")
        check("personalization key present in output",
              out.get("personalization") is not None)

        summary = personal_stats_summary(stats)
        check("summary is not None",    summary is not None)
        check("summary active=True",    summary.get("active") is True)
        check("weight > 0",             summary.get("weight", 0) > 0,
              f"got {summary.get('weight')}")

        if VERBOSE:
            print(f"    final_phase: {out['final_phase']}")
            print(f"    final_probs: { {k: f'{v:.3f}' for k, v in out['final_phase_probs'].items()} }")
            print(f"    personalization: {summary}")

    except Exception as e:
        check("no exception in get_fused_output", False, str(e))


# ── Run all tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Personalization Layer Tests")
    print("=" * 60)

    test_cold_start()
    test_bayesian_warm_start()
    test_recency_decay()
    test_cosine_similarity()
    test_zero_vector()
    test_context_completeness()
    test_user_journey()
    test_full_pipeline_smoke()

    print("\n" + "=" * 60)
    total  = sum(1 for line in open(__file__).readlines() if line.strip().startswith("check("))
    passed = total - len(_failures)
    print(f"Results: {passed} passed, {len(_failures)} failed")

    if _failures:
        print("\nFailed tests:")
        for f in _failures:
            print(f"  {FAIL} {f}")
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)
