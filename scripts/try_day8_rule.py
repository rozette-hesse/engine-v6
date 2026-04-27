#!/usr/bin/env python3
"""
Measure the impact of extending the menstrual-continuation rule to day 8.

Current day 7 rule: Menstrual only if flow is substantial AND yesterday was
also substantial. Day 8+ returns False unconditionally.

Proposal: day 8 follows the same 2-day-substantial rule as day 7.

Prints TP/FP delta and the resulting overall accuracy change on the held-out
set. Does NOT modify production code — patches the function in-process and
reverts via unload.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning)

from cycle import fusion as fusion_mod
from cycle.fusion import get_fused_output, _flow_category
from cycle.layer2.features_v3 import (
    SUPPORTED_SYMPTOMS,
    severities_to_binary_symptoms,
    to_ordinal,
)

CSV_PATH      = ROOT / "data" / "mcphases_with_estimated_cervical_mucus_v3.csv"
HELD_OUT_FILE = ROOT / "scripts" / "held_out_participants.json"
BASE_DATE     = datetime(2022, 1, 1)


# ── Proposed rule replacement ─────────────────────────────────────────────────
def _is_ongoing_menstrual_with_day8(layer1, flow, recent_daily_logs):
    """Current rules + extend the day-7 substantial-2-day rule to day 8."""
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


def _row_severities(row):
    return {s: to_ordinal(row.get(s)) for s in SUPPORTED_SYMPTOMS}


def _row_flow(row):
    v = row.get("flow_volume")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v).strip()


def _row_mucus(row):
    v = row.get("cervical_mucus_estimated_type_v3") or "unknown"
    return {"egg_white": "eggwhite"}.get(str(v), str(v))


def replay(grp):
    grp = grp.sort_values("day_in_study").reset_index(drop=True)
    grp["date"] = grp["day_in_study"].apply(
        lambda d: (BASE_DATE + timedelta(days=int(d) - 1)).strftime("%Y-%m-%d")
    )

    known_starts = []
    prev_men = False
    history_buffer = []
    rows_out = []

    for _, row in grp.iterrows():
        truth  = str(row["phase"])
        today  = row["date"]
        is_men = truth == "Menstrual"
        is_day1 = is_men and not prev_men
        if is_day1:
            known_starts.append(today)
        if not known_starts:
            prev_men = is_men
            continue

        severities = _row_severities(row)
        recent = [
            {
                "symptoms":       severities_to_binary_symptoms(h["severities"]),
                "cervical_mucus": h["mucus"],
                "appetite":       h["appetite"],
                "exerciselevel":  h["exerciselevel"],
                "flow":           h["flow"],
                "symptom_severities": h["severities"],
            }
            for h in history_buffer[-3:]
        ]

        out = get_fused_output(
            period_starts=list(known_starts),
            symptoms=severities_to_binary_symptoms(severities),
            cervical_mucus=_row_mucus(row),
            appetite=to_ordinal(row.get("appetite")),
            exerciselevel=to_ordinal(row.get("exerciselevel")),
            period_start_logged=is_day1,
            recent_daily_logs=recent,
            today=today,
            flow=_row_flow(row),
            symptom_severities=severities,
        )

        rows_out.append((truth, out["final_phase"]))

        history_buffer.append({
            "severities":    severities,
            "appetite":      to_ordinal(row.get("appetite")),
            "exerciselevel": to_ordinal(row.get("exerciselevel")),
            "flow":          _row_flow(row),
            "mucus":         _row_mucus(row),
        })
        prev_men = is_men

    return rows_out


def score(results):
    from collections import Counter
    conf = Counter()
    for truth, pred in results:
        conf[(truth, pred)] += 1
    n = len(results)
    correct = sum(conf[(t, t)] for t in {t for t, _ in results})
    acc = correct / n if n else 0.0
    return acc, conf


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-2022", action="store_true",
                        help="Score on the full 2022 wave (not just held-out).")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)
    with open(HELD_OUT_FILE) as f:
        ho = set(json.load(f)["held_out_participant_ids"])
    if args.all_2022:
        df = df[df["study_interval"] == 2022]
        print(f"Scoring full 2022 wave: {df['id'].nunique()} PIDs, {len(df)} rows\n")
    else:
        df = df[(df["study_interval"] == 2022) & (df["id"].isin(ho))]
        print(f"Scoring held-out only: {sorted(ho)}\n")

    # ── Baseline ──────────────────────────────────────────────────────────────
    results_base = []
    for _, g in df.groupby("id"):
        results_base.extend(replay(g))
    acc_base, conf_base = score(results_base)

    # ── Patched: day-8 rule active ────────────────────────────────────────────
    original = fusion_mod._is_ongoing_menstrual
    fusion_mod._is_ongoing_menstrual = _is_ongoing_menstrual_with_day8
    try:
        results_patch = []
        for _, g in df.groupby("id"):
            results_patch.extend(replay(g))
    finally:
        fusion_mod._is_ongoing_menstrual = original
    acc_patch, conf_patch = score(results_patch)

    print(f"Held-out rows scored: {len(results_base)}")
    print(f"\nBaseline accuracy:           {acc_base*100:.2f}%")
    print(f"With day-8 extension:        {acc_patch*100:.2f}%")
    print(f"Δ: {(acc_patch-acc_base)*100:+.2f}pp")

    # Which predictions changed?
    changes_help = changes_hurt = unchanged = 0
    for (t_b, p_b), (t_p, p_p) in zip(results_base, results_patch):
        assert t_b == t_p
        if p_b == p_p:
            unchanged += 1
        elif p_p == t_p and p_b != t_b:
            changes_help += 1
        elif p_p != t_p and p_b == t_b:
            changes_hurt += 1

    print(f"\nPredictions changed: help={changes_help}  hurt={changes_hurt}  unchanged={unchanged}")
    print("\nConfusion-matrix delta (patch - base), cells with non-zero change:")
    phases = ["Menstrual", "Follicular", "Fertility", "Luteal"]
    for t in phases:
        for p in phases:
            d = conf_patch[(t, p)] - conf_base[(t, p)]
            if d:
                print(f"  truth={t:10s} pred={p:10s} : {d:+d}")


if __name__ == "__main__":
    main()
