#!/usr/bin/env python3
"""
Compute data-derived population priors for personalization.

Replaces the hand-tuned _HARDCODED_PRIORS dict in cycle/personalization.py with
empirical per-phase symptom means derived from the 2022 training wave, excluding
the held-out participants reserved for final evaluation.

Output
------
Writes artifacts/population_phase_means.json with the schema expected by
cycle.personalization.load_population_priors():

    {
        "Follicular": {<symptom>: mean_severity_0_5, ...},
        "Fertility":  {...},
        "Luteal":     {...},
        "_meta": { ...provenance fields... }
    }

_meta is ignored by load_population_priors (it validates that all phases and
symptoms are present; extra keys are passed through) but gives us an audit trail.

Why 2022 only, held-out excluded
--------------------------------
- 2024 wave has 99.4% symptom missingness (see train_layer2_v4.py) so it would
  bias the means toward zero without adding signal.
- Held-out participants (see scripts/held_out_participants.json) must not leak
  into any training artifact — population priors are part of the training setup.

Run
---
  python scripts/compute_population_priors.py

This file is idempotent. Re-run after dataset changes or a held-out reshuffle
(don't reshuffle — see held_out_participants.json note).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cycle.config import ARTIFACTS_DIR, NON_MENSTRUAL_PHASES, SUPPORTED_SYMPTOMS
from cycle.layer2.features_v3 import to_ordinal

DATA_CSV      = ROOT / "data" / "mcphases_with_estimated_cervical_mucus_v3.csv"
HELD_OUT_FILE = ROOT / "scripts" / "held_out_participants.json"
# NOTE: filename deliberately NOT the one auto-loaded by
# cycle.personalization.load_population_priors() (which looks for
# "population_phase_means.json"). The empirical means on the 2022 wave are
# near-parallel across phases (pairwise cosine > 0.99) due to symptomatic-
# logging bias, so using them as the Bayesian prior would silently suppress
# personalization for every cold-start user (discriminability gate triggers).
# The clinically-informed _HARDCODED_PRIORS in personalization.py are a
# better starting point until we have richer-logging data. Keep this artifact
# around for audit / future analysis. To activate: rename or symlink to
# "population_phase_means.json".
OUTPUT_FILE   = ARTIFACTS_DIR / "population_phase_means_empirical_2022.json"


def load_held_out_pids() -> set:
    with open(HELD_OUT_FILE) as f:
        return set(json.load(f)["held_out_participant_ids"])


def compute_priors(df: pd.DataFrame, held_out_pids: set) -> Dict:
    """
    Per-phase ordinal-symptom means on 2022-wave rows, excluding held-out PIDs
    and rows where all symptoms are null (pure logging gaps add zero bias, but
    this filter makes the n_days_used number honest).
    """
    pool = df[(df["study_interval"] == 2022) & (~df["id"].isin(held_out_pids))].copy()

    # Convert ordinal strings to 0..5 integers
    for sym in SUPPORTED_SYMPTOMS:
        pool[sym] = pool[sym].apply(to_ordinal)

    any_logged = pool[SUPPORTED_SYMPTOMS].sum(axis=1) > 0
    print(f"  Pool       : {len(pool)} rows | "
          f"{pool['id'].nunique()} participants (2022, held-out excluded)")
    print(f"  Any symptom: {int(any_logged.sum())} rows "
          f"({100 * any_logged.mean():.1f}%)")

    priors: Dict[str, Dict[str, float]] = {}
    coverage: Dict[str, int] = {}

    for phase in NON_MENSTRUAL_PHASES:
        phase_rows = pool[(pool["phase"] == phase) & any_logged]
        coverage[phase] = len(phase_rows)
        if len(phase_rows) == 0:
            raise RuntimeError(f"No rows for phase {phase!r} — cannot compute prior.")
        means = {sym: float(phase_rows[sym].mean()) for sym in SUPPORTED_SYMPTOMS}
        priors[phase] = means

    priors["_meta"] = {
        "generated_at":      datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_csv":        str(DATA_CSV.relative_to(ROOT)),
        "study_interval":    2022,
        "held_out_excluded": sorted(held_out_pids),
        "n_participants":    int(pool["id"].nunique()),
        "n_rows_any_symptom": int(any_logged.sum()),
        "rows_by_phase":     coverage,
        "notes": (
            "Means are on the 0-5 ordinal severity scale (to_ordinal). "
            "Rows with no symptoms logged are excluded so means reflect the "
            "typical severity WHEN a user logs, not diluted-by-missingness."
        ),
    }
    return priors


def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
    dot    = sum(a[s] * b[s] for s in SUPPORTED_SYMPTOMS)
    norm_a = float(np.sqrt(sum(a[s] ** 2 for s in SUPPORTED_SYMPTOMS)))
    norm_b = float(np.sqrt(sum(b[s] ** 2 for s in SUPPORTED_SYMPTOMS)))
    return 1.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


def report_discriminability(priors: Dict) -> None:
    print("\nPairwise cosine similarity between phase means:")
    max_cos = 0.0
    for i, p1 in enumerate(NON_MENSTRUAL_PHASES):
        for p2 in NON_MENSTRUAL_PHASES[i + 1:]:
            c = _cos(priors[p1], priors[p2])
            max_cos = max(max_cos, c)
            print(f"  {p1:11s} vs {p2:11s}: cos = {c:.4f}")
    gate_thresh = 0.97
    if max_cos >= gate_thresh:
        print(f"\n  ⚠ Max pairwise cosine {max_cos:.4f} >= gate threshold {gate_thresh}.")
        print("  Using these as Bayesian priors would trigger the discriminability")
        print("  gate for every cold-start user (n_real=0 → posterior = prior).")
        print("  Personalization would be silently suppressed until enough real")
        print("  data shifts per-phase means apart. Keeping the clinically-informed")
        print("  _HARDCODED_PRIORS in personalization.py is safer until logging")
        print("  coverage improves.")
    else:
        print(f"\n  Max pairwise cosine {max_cos:.4f} < gate threshold {gate_thresh}. OK.")


def pretty_print(priors: Dict) -> None:
    header = f"  {'':13s}" + "".join(f"{s:>14s}" for s in SUPPORTED_SYMPTOMS)
    print(header)
    for phase in NON_MENSTRUAL_PHASES:
        row = f"  {phase:13s}" + "".join(f"{priors[phase][s]:>14.2f}" for s in SUPPORTED_SYMPTOMS)
        print(row)


def main() -> int:
    print(f"Loading data: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    held_out = load_held_out_pids()
    print(f"Held-out PIDs (excluded): {sorted(held_out)}")

    priors = compute_priors(df, held_out)

    print("\nData-derived per-phase symptom means (0-5 scale):")
    pretty_print(priors)
    report_discriminability(priors)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(priors, f, indent=2, sort_keys=False)
    print(f"\nWrote {OUTPUT_FILE}")
    print("NOTE: this file is NOT auto-loaded. The production default remains the")
    print("clinically-informed _HARDCODED_PRIORS in cycle/personalization.py.")
    print("To activate these empirical priors, rename/symlink to "
          "'population_phase_means.json'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
