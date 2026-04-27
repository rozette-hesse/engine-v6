#!/usr/bin/env python3
"""
Cycle Engine — Dataset-Level Accuracy Evaluation
=================================================
Evaluates the full fusion pipeline against the labelled clinical dataset
(mcphases_with_estimated_cervical_mucus_v3.csv, 2022 wave only).

Each participant's days are replayed in order, passing the previous 1-2
days as ``recent_daily_logs`` so the menstrual-continuation detector and
the rolling history features have the context they need.

Run:
  python scripts/evaluate_cycle.py
  python scripts/evaluate_cycle.py --all      # include 2024 wave
  python scripts/evaluate_cycle.py --held-out # held-out participants only

Output
------
  Per-phase precision / recall / F1
  Overall accuracy (all phases, non-menstrual only)
  Confusion matrix
  Per-mode breakdown
"""

import sys
import os
import argparse
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

from cycle.fusion import get_fused_output
from cycle.config import PHASES
from cycle.layer2.features_v3 import to_ordinal, severities_to_binary_symptoms, SUPPORTED_SYMPTOMS

CSV_PATH       = ROOT / "data" / "mcphases_with_estimated_cervical_mucus_v3.csv"
HELD_OUT_FILE  = ROOT / "scripts" / "held_out_participants.json"
BASE_DATE      = datetime(2022, 1, 1)


# ── Data helpers ──────────────────────────────────────────────────────────────

def row_to_severities(row) -> dict:
    return {sym: to_ordinal(row.get(sym)) for sym in SUPPORTED_SYMPTOMS}


def row_to_flow(row) -> str:
    val = row.get("flow_volume")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "none"
    s = str(val).strip()
    return "none" if s.lower() in {"", "nan", "not at all"} else s


def row_to_appetite(row) -> int:
    return to_ordinal(row.get("appetite"))


def row_to_exerciselevel(row) -> int:
    return to_ordinal(row.get("exerciselevel"))


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame, label: str) -> dict:
    """
    Replay each participant day-by-day and collect (truth, predicted, mode) tuples.
    Returns a dict of lists ready for metric computation.
    """
    results = {"truth": [], "pred": [], "mode": [], "pid": []}
    skipped = 0

    for pid, grp in df.groupby("id", sort=True):
        grp = grp.sort_values("day_in_study").reset_index(drop=True)
        grp["_date"] = grp["day_in_study"].apply(
            lambda d: (BASE_DATE + timedelta(days=int(d) - 1)).strftime("%Y-%m-%d")
        )

        known_starts   = []
        history_buffer = []   # list of {severities, cervical_mucus, appetite, exerciselevel, flow}
        prev_menstrual = False

        for _, row in grp.iterrows():
            phase_truth = str(row.get("phase", ""))
            today       = str(row["_date"])

            if phase_truth in ("nan", "None", "") or pd.isna(row.get("phase")):
                # Missing label — update history but don't score
                severities    = row_to_severities(row)
                appetite_val  = row_to_appetite(row)
                exercise_val  = row_to_exerciselevel(row)
                flow_val      = row_to_flow(row)
                history_buffer.append({
                    "severities":    severities,
                    "cervical_mucus": "unknown",
                    "appetite":      appetite_val,
                    "exerciselevel": exercise_val,
                    "flow":          flow_val,
                })
                prev_menstrual = (phase_truth == "Menstrual")
                continue

            is_men  = (phase_truth == "Menstrual")
            is_day1 = is_men and not prev_menstrual

            if is_day1:
                known_starts.append(today)

            if not known_starts:
                skipped += 1
                prev_menstrual = is_men
                continue

            severities    = row_to_severities(row)
            appetite_val  = row_to_appetite(row)
            exercise_val  = row_to_exerciselevel(row)
            flow_val      = row_to_flow(row)

            recent_daily_logs = [
                {
                    "symptoms":           severities_to_binary_symptoms(h["severities"]),
                    "symptom_severities": h["severities"],   # used by v5 for ordinal lags
                    "cervical_mucus":     h["cervical_mucus"],
                    "appetite":           h["appetite"],
                    "exerciselevel":      h["exerciselevel"],
                    "flow":               h["flow"],
                }
                for h in history_buffer[-4:]   # 4 prior days — v5 uses all 4; v4 uses last 2
            ]

            out = get_fused_output(
                period_starts=list(known_starts),
                symptoms=severities_to_binary_symptoms(severities),
                symptom_severities=severities,
                cervical_mucus="unknown",   # mucus in CSV is label-derived, not user-observed
                appetite=appetite_val,
                exerciselevel=exercise_val,
                flow=flow_val,
                today=today,
                recent_daily_logs=recent_daily_logs,
            )

            results["truth"].append(phase_truth)
            results["pred"].append(out["final_phase"])
            results["mode"].append(out["mode"])
            results["pid"].append(pid)

            history_buffer.append({
                "severities":    severities,
                "cervical_mucus": "unknown",
                "appetite":      appetite_val,
                "exerciselevel": exercise_val,
                "flow":          flow_val,
            })
            prev_menstrual = is_men

    print(f"  Scored: {len(results['truth'])} rows  |  Skipped (no history): {skipped}")
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: dict) -> None:
    truth = results["truth"]
    pred  = results["pred"]
    modes = results["mode"]

    n = len(truth)
    if n == 0:
        print("No rows to score.")
        return

    # ── Overall accuracy ──────────────────────────────────────────────────────
    overall_correct = sum(t == p for t, p in zip(truth, pred))
    print(f"\n  Overall accuracy:          {overall_correct/n*100:.1f}%  ({overall_correct}/{n})")

    non_men_mask = [t != "Menstrual" for t in truth]
    nm_truth = [t for t, m in zip(truth, non_men_mask) if m]
    nm_pred  = [p for p, m in zip(pred,  non_men_mask) if m]
    nm_correct = sum(t == p for t, p in zip(nm_truth, nm_pred))
    nm_n = len(nm_truth)
    if nm_n:
        print(f"  Non-menstrual accuracy:    {nm_correct/nm_n*100:.1f}%  ({nm_correct}/{nm_n})")

    # ── Per-phase metrics ─────────────────────────────────────────────────────
    print(f"\n  {'Phase':<14} {'True N':>7} {'Pred N':>7} {'Recall':>8} {'Precision':>10} {'F1':>7}")
    print(f"  {'─'*14} {'─'*7} {'─'*7} {'─'*8} {'─'*10} {'─'*7}")

    for phase in PHASES:
        tp = sum(t == phase and p == phase for t, p in zip(truth, pred))
        fn = sum(t == phase and p != phase for t, p in zip(truth, pred))
        fp = sum(t != phase and p == phase for t, p in zip(truth, pred))
        true_n = tp + fn
        pred_n = tp + fp
        recall    = tp / true_n if true_n else 0.0
        precision = tp / pred_n if pred_n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"  {phase:<14} {true_n:>7} {pred_n:>7} {recall*100:>7.1f}% {precision*100:>9.1f}% {f1*100:>6.1f}%")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    print(f"\n  Confusion matrix (rows=truth, cols=predicted):")
    print(f"  {'':14}", end="")
    for p in PHASES:
        print(f"  {p[:8]:>8}", end="")
    print()
    for t_phase in PHASES:
        print(f"  {t_phase:<14}", end="")
        for p_phase in PHASES:
            count = sum(t == t_phase and p == p_phase for t, p in zip(truth, pred))
            print(f"  {count:>8}", end="")
        print()

    # ── Mode breakdown ────────────────────────────────────────────────────────
    mode_counts: dict = defaultdict(int)
    mode_correct: dict = defaultdict(int)
    for t, p, m in zip(truth, pred, modes):
        mode_counts[m] += 1
        if t == p:
            mode_correct[m] += 1

    print(f"\n  Fusion mode breakdown:")
    print(f"  {'Mode':<35} {'N':>6} {'Acc':>7}")
    print(f"  {'─'*35} {'─'*6} {'─'*7}")
    for mode in sorted(mode_counts, key=mode_counts.get, reverse=True):
        mn = mode_counts[mode]
        mc = mode_correct[mode]
        print(f"  {mode:<35} {mn:>6} {mc/mn*100:>6.1f}%")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cycle engine accuracy evaluation")
    parser.add_argument("--all",      action="store_true", help="Include 2024 wave (high missingness)")
    parser.add_argument("--held-out", action="store_true", help="Evaluate held-out participants only")
    args = parser.parse_args()

    print(f"\nCycle Engine — Accuracy Evaluation\n{'─'*65}")
    print(f"  Dataset: {CSV_PATH.name}")

    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded:  {len(df):,} rows | {df['id'].nunique()} participants")

    held_out_pids: set = set()
    if HELD_OUT_FILE.exists():
        import json
        with open(HELD_OUT_FILE) as f:
            held_out_pids = set(json.load(f).get("held_out_participant_ids", []))
        print(f"  Held-out participants: {sorted(held_out_pids)}")

    # Filter wave
    if not args.all:
        df = df[df["study_interval"] == 2022]
        print(f"  Using 2022 wave: {len(df):,} rows")

    # Filter participant split
    if args.held_out:
        if not held_out_pids:
            print("  WARNING: no held_out_participants.json found — evaluating all participants")
        else:
            df = df[df["id"].isin(held_out_pids)]
            print(f"  Held-out only: {len(df):,} rows | {df['id'].nunique()} participants")
    elif held_out_pids:
        df = df[~df["id"].isin(held_out_pids)]
        print(f"  Excluding held-out: {len(df):,} rows | {df['id'].nunique()} participants")

    print()
    results = evaluate(df, label="")
    print()
    compute_metrics(results)
    print()


if __name__ == "__main__":
    main()
