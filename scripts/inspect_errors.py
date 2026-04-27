#!/usr/bin/env python3
"""
Drill into held-out errors to find rule-tweakable vs irreducible clusters.

Replays each held-out participant chronologically (same as evaluate_cycle.py)
and prints the cycle_day / fusion mode / phase_probs for every error,
grouped by the (truth → predicted) pair.

Focus on the two largest error clusters observed in the baseline:
    Menstrual   → Follicular  (n=22)
    Fertility   → Luteal      (n=20)
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning)

from cycle.fusion import get_fused_output
from cycle.layer2.features_v3 import (
    SUPPORTED_SYMPTOMS,
    severities_to_binary_symptoms,
    to_ordinal,
)

CSV_PATH       = ROOT / "data" / "mcphases_with_estimated_cervical_mucus_v3.csv"
HELD_OUT_FILE  = ROOT / "scripts" / "held_out_participants.json"
BASE_DATE      = datetime(2022, 1, 1)


def _row_severities(row) -> dict:
    return {s: to_ordinal(row.get(s)) for s in SUPPORTED_SYMPTOMS}


def _row_flow(row):
    v = row.get("flow_volume")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v).strip()


def _row_mucus(row) -> str:
    v = row.get("cervical_mucus_estimated_type_v3") or "unknown"
    return {"egg_white": "eggwhite"}.get(str(v), str(v))


def replay_participant(grp):
    grp = grp.sort_values("day_in_study").reset_index(drop=True)
    grp["date"] = grp["day_in_study"].apply(
        lambda d: (BASE_DATE + timedelta(days=int(d) - 1)).strftime("%Y-%m-%d")
    )

    known_starts = []
    prev_men = False
    history_buffer = []
    errors = []

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

        pred = out["final_phase"]
        if pred != truth:
            errors.append({
                "pid":        int(row["id"]),
                "date":       today,
                "truth":      truth,
                "pred":       pred,
                "cycle_day":  out["layer1"].get("cycle_day"),
                "mode":       out.get("mode"),
                "final_probs": out.get("final_phase_probs"),
                "l1_top":      (out.get("layer1") or {}).get("top_phase"),
                "l2_top":      ((out.get("layer2") or {}).get("top_phase") or None),
                "flow":        _row_flow(row),
            })

        history_buffer.append({
            "severities":    severities,
            "appetite":      to_ordinal(row.get("appetite")),
            "exerciselevel": to_ordinal(row.get("exerciselevel")),
            "flow":          _row_flow(row),
            "mucus":         _row_mucus(row),
        })
        prev_men = is_men

    return errors


def main():
    df = pd.read_csv(CSV_PATH)
    with open(HELD_OUT_FILE) as f:
        ho = set(json.load(f)["held_out_participant_ids"])
    df = df[(df["study_interval"] == 2022) & (df["id"].isin(ho))]
    print(f"Held-out participants: {sorted(ho)}, {len(df)} rows\n")

    all_errors = []
    for _, grp in df.groupby("id"):
        all_errors.extend(replay_participant(grp))

    print(f"Total errors: {len(all_errors)}\n")

    # Group by (truth, pred)
    bucket = defaultdict(list)
    for e in all_errors:
        bucket[(e["truth"], e["pred"])].append(e)

    print(f"{'Truth':12s} {'Pred':12s}  N")
    for (t, p), errs in sorted(bucket.items(), key=lambda x: -len(x[1])):
        print(f"  {t:12s} {p:12s} {len(errs)}")

    # Deep-dive the two largest clusters
    for target in [("Menstrual", "Follicular"), ("Fertility", "Luteal"),
                   ("Follicular", "Menstrual"), ("Fertility", "Follicular")]:
        errs = bucket.get(target, [])
        if not errs:
            continue
        print(f"\n─── {target[0]} → {target[1]} ({len(errs)} errors) ───")

        # Cycle-day histogram
        days = Counter(e["cycle_day"] for e in errs)
        print(f"  Cycle-day histogram:")
        for d, n in sorted(days.items(), key=lambda x: (x[0] is None, x[0] or 0)):
            bar = "█" * n
            print(f"    day {str(d):>4s}: {n:3d} {bar}")

        # Mode histogram
        modes = Counter(e["mode"] for e in errs)
        print(f"  Fusion mode histogram:")
        for m, n in modes.most_common():
            print(f"    {m:<32s} {n}")

        # Flow histogram (only for the Menstrual-containing buckets)
        if "Menstrual" in target:
            flows = Counter(e["flow"] for e in errs)
            print(f"  Flow-value histogram:")
            for f, n in flows.most_common():
                print(f"    {str(f):<30s} {n}")


if __name__ == "__main__":
    main()
