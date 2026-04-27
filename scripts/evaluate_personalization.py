#!/usr/bin/env python3
"""
Personalization — Progressive Held-Out Evaluation
==================================================
Does the personalization layer actually improve accuracy on real-world data,
or is it only mathematically reasonable?

This script simulates exactly what happens in production:

  1. For each held-out participant, replay their days in chronological order.
  2. On each day, build personal_stats from the participant's accumulated
     history (logs from earlier days — no future leakage).
  3. Predict the phase twice:
        BASELINE     — no personalization   (personal_stats=None)
        TREATMENT    — with personalization (personal_stats passed in)
  4. Append today's observation to history with its (predicted or truth) phase
     label, and carry on.
  5. Compute per-phase F1 and overall accuracy for both arms and report the
     delta so we can decide whether the personalization layer is a net win.

The only reliable way to answer "does this help?" — the mechanisms are
mathematically reasonable but only empirical evaluation on unseen users
tells us whether they are net-positive in aggregate.

Run
---
  python scripts/evaluate_personalization.py              # held-out, predicted history
  python scripts/evaluate_personalization.py --oracle     # use truth labels in history
  python scripts/evaluate_personalization.py --all        # all participants
"""

import sys
import os
import argparse
import importlib
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# Note: cycle.personalization reads PERSONAL_WEIGHT_MAX / PERSONAL_WEIGHT_RAMP
# env vars at import time. For sweep mode we set them before importing, then
# reload between runs so the fresh values take effect.
from cycle.fusion import get_fused_output
from cycle.config import PHASES, NON_MENSTRUAL_PHASES
import cycle.personalization as personalization_mod
from cycle.personalization import compute_personal_stats
from cycle.layer2.features_v3 import to_ordinal, severities_to_binary_symptoms, SUPPORTED_SYMPTOMS

CSV_PATH       = ROOT / "data" / "mcphases_with_estimated_cervical_mucus_v3.csv"
HELD_OUT_FILE  = ROOT / "scripts" / "held_out_participants.json"
BASE_DATE      = datetime(2022, 1, 1)


# ── Data helpers (same as evaluate_cycle) ─────────────────────────────────────

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

def _synthesize_prior_history(grp: pd.DataFrame, n_copies: int) -> list:
    """
    Build synthetic prior history by replicating the participant's own labeled
    data backwards in time.

    For n_copies=1, we prepend one full copy of the participant's labeled
    period (typically 90 days) ending one day before their real start. This
    simulates "they already used the app for another 3 months before the
    evaluation window" without inventing symptom patterns — their own real
    patterns are the most plausible prior for them.

    Per-phase symptom MEANS are unchanged (same data replicated), but n_real
    grows by n_copies×, which pushes the personalization blend weight up:

        default ramp: weight = 0.35 × n / (n+30)
        n=15 real   → weight = 0.12
        n=30 real   → weight = 0.18
        n=60 real   → weight = 0.23 (+6mo)
        n=90 real   → weight = 0.26 (+9mo)
    """
    if n_copies <= 0:
        return []

    # Use only rows with valid phase labels and symptom data
    template_rows = []
    first_real_date = None
    for _, row in grp.sort_values("day_in_study").iterrows():
        phase_truth = str(row.get("phase", ""))
        if phase_truth in ("nan", "None", "") or pd.isna(row.get("phase")):
            continue
        date_str = (BASE_DATE + timedelta(days=int(row["day_in_study"]) - 1)).strftime("%Y-%m-%d")
        if first_real_date is None:
            first_real_date = datetime.strptime(date_str, "%Y-%m-%d")
        template_rows.append({
            "phase":      phase_truth,
            "severities": row_to_severities(row),
            "flow":       row_to_flow(row),
            "appetite":   row_to_appetite(row),
            "exerciselevel": row_to_exerciselevel(row),
        })

    if not template_rows or first_real_date is None:
        return []

    synthetic = []
    n_template = len(template_rows)
    # Fill `n_copies × n_template` days before first_real_date
    # Earliest synthetic day first (oldest-first for history chronology).
    total_days = n_copies * n_template
    for day_offset in range(total_days, 0, -1):
        # day_offset=total_days → earliest; day_offset=1 → one day before real start
        synth_date = first_real_date - timedelta(days=day_offset)
        # Cycle through template: take day (total_days - day_offset) % n_template
        template_idx = (total_days - day_offset) % n_template
        t = template_rows[template_idx]
        synthetic.append({
            "date":               synth_date.strftime("%Y-%m-%d"),
            "symptom_severities": t["severities"],
            "final_phase":        t["phase"],
            "cycle_day":          None,
            "period_starts":      [],
        })
    return synthetic


def evaluate_participant(grp: pd.DataFrame, pid, use_oracle: bool,
                         pre_history_cycles: int = 0) -> dict:
    """
    Replay one participant's days chronologically, predicting twice per day:
    once without personalization (baseline) and once with (treatment).

    ``pre_history_cycles`` pre-seeds the personal-stats history with N copies
    of the participant's own labeled data, simulating a user who has been
    logging for (1 + N) × the evaluation window before the test window starts.
    Has no effect on the baseline arm.

    Returns a dict of lists indexed by arm:
        {"baseline":  {"truth": [...], "pred": [...]},
         "treatment": {"truth": [...], "pred": [...]},
         "pid":       [...],
         "day_idx":   [...]}
    """
    grp = grp.sort_values("day_in_study").reset_index(drop=True)
    grp["_date"] = grp["day_in_study"].apply(
        lambda d: (BASE_DATE + timedelta(days=int(d) - 1)).strftime("%Y-%m-%d")
    )

    known_starts    = []
    recent_buffer   = []   # last ~4 days for recent_daily_logs
    history_logs    = _synthesize_prior_history(grp, pre_history_cycles)
    prev_menstrual  = False

    out_rows = {
        "baseline":  {"truth": [], "pred": [], "mode": []},
        "treatment": {"truth": [], "pred": [], "mode": []},
        "pid":       [],
        "day_idx":   [],
    }

    for _, row in grp.iterrows():
        phase_truth = str(row.get("phase", ""))
        today       = str(row["_date"])

        severities    = row_to_severities(row)
        appetite_val  = row_to_appetite(row)
        exercise_val  = row_to_exerciselevel(row)
        flow_val      = row_to_flow(row)

        # Unlabelled rows: update history so future personal_stats sees them,
        # but skip scoring this day.
        if phase_truth in ("nan", "None", "") or pd.isna(row.get("phase")):
            recent_buffer.append({
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

        # Can't run L1 without any known period start — skip until we have one.
        if not known_starts:
            prev_menstrual = is_men
            continue

        recent_daily_logs = [
            {
                "symptoms":           severities_to_binary_symptoms(h["severities"]),
                "symptom_severities": h["severities"],
                "cervical_mucus":     h["cervical_mucus"],
                "appetite":           h["appetite"],
                "exerciselevel":      h["exerciselevel"],
                "flow":               h["flow"],
            }
            for h in recent_buffer[-4:]
        ]

        shared = dict(
            period_starts=list(known_starts),
            symptoms=severities_to_binary_symptoms(severities),
            symptom_severities=severities,
            cervical_mucus="unknown",
            appetite=appetite_val,
            exerciselevel=exercise_val,
            flow=flow_val,
            today=today,
            recent_daily_logs=recent_daily_logs,
        )

        # ── Baseline arm: no personal_stats ───────────────────────────────────
        base_out = get_fused_output(**shared, personal_stats=None)

        # ── Treatment arm: personal_stats from accumulated history ────────────
        personal_stats = compute_personal_stats(history_logs, reference_date=today)
        treat_out      = get_fused_output(**shared, personal_stats=personal_stats)

        out_rows["baseline"]["truth"].append(phase_truth)
        out_rows["baseline"]["pred"].append(base_out["final_phase"])
        out_rows["baseline"]["mode"].append(base_out["mode"])
        out_rows["treatment"]["truth"].append(phase_truth)
        out_rows["treatment"]["pred"].append(treat_out["final_phase"])
        out_rows["treatment"]["mode"].append(treat_out["mode"])
        out_rows["pid"].append(pid)
        out_rows["day_idx"].append(len(out_rows["pid"]))

        # ── Append today to history with its phase label ──────────────────────
        # Oracle mode: seed history with ground truth (upper bound — what the
        # personalization layer could achieve with perfect phase labels).
        # Default: seed with the prediction from the treatment arm (production
        # behaviour — history reflects what the user would have seen).
        seed_phase = phase_truth if use_oracle else treat_out["final_phase"]
        history_logs.append({
            "date":               today,
            "symptom_severities": severities,
            "final_phase":        seed_phase,
            "cycle_day":          base_out.get("layer1", {}).get("cycle_day"),
            "period_starts":      list(known_starts),
        })

        recent_buffer.append({
            "severities":    severities,
            "cervical_mucus": "unknown",
            "appetite":      appetite_val,
            "exerciselevel": exercise_val,
            "flow":          flow_val,
        })
        prev_menstrual = is_men

    return out_rows


def evaluate(df: pd.DataFrame, use_oracle: bool,
             pre_history_cycles: int = 0) -> dict:
    """Aggregate across participants."""
    agg = {
        "baseline":  {"truth": [], "pred": [], "mode": []},
        "treatment": {"truth": [], "pred": [], "mode": []},
        "pid":       [],
        "day_idx":   [],
    }
    for pid, grp in df.groupby("id", sort=True):
        rows = evaluate_participant(grp, pid, use_oracle=use_oracle,
                                    pre_history_cycles=pre_history_cycles)
        for arm in ("baseline", "treatment"):
            agg[arm]["truth"].extend(rows[arm]["truth"])
            agg[arm]["pred"].extend(rows[arm]["pred"])
            agg[arm]["mode"].extend(rows[arm]["mode"])
        agg["pid"].extend(rows["pid"])
        agg["day_idx"].extend(rows["day_idx"])

    n_scored = len(agg["pid"])
    print(f"  Scored: {n_scored} rows across {df['id'].nunique()} participants")
    return agg


# ── Metrics ───────────────────────────────────────────────────────────────────

def _arm_metrics(truth, pred):
    """Return dict of per-phase and overall metrics for one arm."""
    n = len(truth)
    overall = sum(t == p for t, p in zip(truth, pred)) / n if n else 0.0

    nm_truth = [t for t in truth if t != "Menstrual"]
    nm_pred  = [p for t, p in zip(truth, pred) if t != "Menstrual"]
    nm_acc = (sum(t == p for t, p in zip(nm_truth, nm_pred)) / len(nm_truth)
              if nm_truth else 0.0)

    per_phase = {}
    for phase in PHASES:
        tp = sum(t == phase and p == phase for t, p in zip(truth, pred))
        fn = sum(t == phase and p != phase for t, p in zip(truth, pred))
        fp = sum(t != phase and p == phase for t, p in zip(truth, pred))
        true_n  = tp + fn
        pred_n  = tp + fp
        recall    = tp / true_n if true_n else 0.0
        precision = tp / pred_n if pred_n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_phase[phase] = {
            "true_n": true_n, "pred_n": pred_n,
            "recall": recall, "precision": precision, "f1": f1,
        }
    return {"overall": overall, "nm_acc": nm_acc, "n": n, "per_phase": per_phase}


def _fmt_delta(new, old, pct=True, width=6):
    d = (new - old) * (100 if pct else 1)
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:>{width}.1f}"


def compute_comparison(results: dict) -> None:
    base  = _arm_metrics(results["baseline"]["truth"],  results["baseline"]["pred"])
    treat = _arm_metrics(results["treatment"]["truth"], results["treatment"]["pred"])
    n = base["n"]

    if n == 0:
        print("No rows to score.")
        return

    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║  BASELINE (no personalization)  vs  TREATMENT (+personal)    ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

    print(f"\n  Overall accuracy:")
    print(f"    Baseline:  {base['overall']*100:5.1f}%")
    print(f"    Treatment: {treat['overall']*100:5.1f}%   Δ {_fmt_delta(treat['overall'], base['overall'])} pp")

    print(f"\n  Non-menstrual accuracy:")
    print(f"    Baseline:  {base['nm_acc']*100:5.1f}%")
    print(f"    Treatment: {treat['nm_acc']*100:5.1f}%   Δ {_fmt_delta(treat['nm_acc'], base['nm_acc'])} pp")

    # ── Per-phase comparison ──────────────────────────────────────────────────
    print(f"\n  Per-phase F1 (baseline → treatment):")
    print(f"  {'Phase':<12}  {'True N':>7}  {'F1 base':>9}  {'F1 treat':>10}  {'Δ F1':>8}")
    print(f"  {'─'*12}  {'─'*7}  {'─'*9}  {'─'*10}  {'─'*8}")
    for phase in PHASES:
        b = base["per_phase"][phase]
        t = treat["per_phase"][phase]
        print(f"  {phase:<12}  {b['true_n']:>7}  "
              f"{b['f1']*100:>8.1f}%  {t['f1']*100:>9.1f}%  "
              f"{_fmt_delta(t['f1'], b['f1'])} pp")

    # ── Disagreements: where did personalization flip the answer? ─────────────
    flips_base_right = 0  # baseline correct, treatment wrong (regressions)
    flips_treat_right = 0 # treatment correct, baseline wrong (improvements)
    same_right  = 0
    same_wrong  = 0
    flip_by_truth = defaultdict(lambda: {"gained": 0, "lost": 0})

    for t, b_pred, t_pred in zip(
        results["baseline"]["truth"],
        results["baseline"]["pred"],
        results["treatment"]["pred"],
    ):
        b_correct = (t == b_pred)
        t_correct = (t == t_pred)
        if b_correct and t_correct:
            same_right += 1
        elif not b_correct and not t_correct:
            same_wrong += 1
        elif t_correct and not b_correct:
            flips_treat_right += 1
            flip_by_truth[t]["gained"] += 1
        else:
            flips_base_right += 1
            flip_by_truth[t]["lost"] += 1

    total_flips = flips_base_right + flips_treat_right
    print(f"\n  Prediction flips (personalization changed the answer):")
    print(f"    Same correct:              {same_right:>5}")
    print(f"    Same wrong:                {same_wrong:>5}")
    print(f"    Gained (treat > base):     {flips_treat_right:>5}")
    print(f"    Lost   (treat < base):     {flips_base_right:>5}")
    print(f"    Net:                       {flips_treat_right - flips_base_right:+d}  "
          f"({(flips_treat_right - flips_base_right) / max(1,n) * 100:+.1f} pp)")
    if total_flips:
        print(f"    Flip accuracy:             {flips_treat_right / total_flips * 100:.1f}% "
              f"({flips_treat_right}/{total_flips})")

    if flip_by_truth:
        print(f"\n  Flip breakdown by true phase:")
        print(f"  {'Phase':<12}  {'Gained':>7}  {'Lost':>6}  {'Net':>6}")
        print(f"  {'─'*12}  {'─'*7}  {'─'*6}  {'─'*6}")
        for phase in PHASES:
            g = flip_by_truth[phase]["gained"]
            l = flip_by_truth[phase]["lost"]
            print(f"  {phase:<12}  {g:>7}  {l:>6}  {g - l:>+6}")


def per_participant_breakdown(results: dict) -> None:
    """Per-participant accuracy table so we can see if any user regressed."""
    truths_b = results["baseline"]["truth"]
    preds_b  = results["baseline"]["pred"]
    preds_t  = results["treatment"]["pred"]
    pids     = results["pid"]

    by_pid: dict = defaultdict(lambda: {"n": 0, "base": 0, "treat": 0})
    for pid, t, bp, tp in zip(pids, truths_b, preds_b, preds_t):
        by_pid[pid]["n"]     += 1
        by_pid[pid]["base"]  += int(t == bp)
        by_pid[pid]["treat"] += int(t == tp)

    print(f"\n  Per-participant accuracy:")
    print(f"  {'PID':>5}  {'N':>4}  {'Baseline':>9}  {'Treatment':>10}  {'Δ':>7}")
    print(f"  {'─'*5}  {'─'*4}  {'─'*9}  {'─'*10}  {'─'*7}")
    for pid in sorted(by_pid):
        r = by_pid[pid]
        b = r["base"]  / r["n"]
        t = r["treat"] / r["n"]
        print(f"  {pid:>5}  {r['n']:>4}  {b*100:>8.1f}%  {t*100:>9.1f}%  "
              f"{_fmt_delta(t, b)} pp")


# ── Entry point ───────────────────────────────────────────────────────────────

def _set_weight_config(max_weight: float, ramp_halfpoint: float) -> None:
    """
    Set blend-weight config by patching module attributes directly.

    We tried importlib.reload() but it didn't propagate to fusion.py's local
    import reliably across iterations. Direct attribute assignment on the
    module object is simpler and verified to change predictions.
    """
    personalization_mod.MAX_PERSONAL_WEIGHT = float(max_weight)
    personalization_mod.RAMP_HALFPOINT      = float(ramp_halfpoint)


def _sweep_summary(truth, pred) -> tuple:
    """Return (overall_acc, nm_acc, f1_by_phase)."""
    n = len(truth)
    overall = sum(t == p for t, p in zip(truth, pred)) / n if n else 0.0
    nm_truth = [t for t in truth if t != "Menstrual"]
    nm_pred  = [p for t, p in zip(truth, pred) if t != "Menstrual"]
    nm_acc = (sum(t == p for t, p in zip(nm_truth, nm_pred)) / len(nm_truth)
              if nm_truth else 0.0)
    f1s = {}
    for phase in PHASES:
        tp = sum(t == phase and p == phase for t, p in zip(truth, pred))
        fn = sum(t == phase and p != phase for t, p in zip(truth, pred))
        fp = sum(t != phase and p == phase for t, p in zip(truth, pred))
        r = tp / (tp + fn) if (tp + fn) else 0.0
        p_ = tp / (tp + fp) if (tp + fp) else 0.0
        f1s[phase] = 2 * p_ * r / (p_ + r) if (p_ + r) else 0.0
    return overall, nm_acc, f1s


def run_sweep(df: pd.DataFrame, use_oracle: bool) -> None:
    """Evaluate across a grid of (max_weight, ramp_halfpoint) settings."""
    # (max_weight, ramp_halfpoint, label)
    configs = [
        (0.35, 30.0, "default        (0.35 × n/(n+30))"),
        (0.50, 20.0, "moderate       (0.50 × n/(n+20))"),
        (0.70, 10.0, "strong         (0.70 × n/(n+10))"),
        (1.00, 5.0,  "aggressive     (1.00 × n/(n+5))"),
    ]

    # ── Baseline (no personalization) — run once ──────────────────────────────
    print("\n  Running baseline (no personalization)...")
    _set_weight_config(0.35, 30.0)
    base_results = evaluate(df, use_oracle=use_oracle)
    base_overall, base_nm, base_f1 = _sweep_summary(
        base_results["baseline"]["truth"], base_results["baseline"]["pred"]
    )

    print(f"\n  Baseline:  overall {base_overall*100:.2f}%  "
          f"non-men {base_nm*100:.2f}%  "
          f"F1s: Fol {base_f1['Follicular']*100:.1f} / "
          f"Fert {base_f1['Fertility']*100:.1f} / "
          f"Lut {base_f1['Luteal']*100:.1f}")

    print(f"\n  {'─'*72}")
    print(f"  {'Config':<36}  {'Overall':>9}  {'ΔOvr':>6}  {'ΔFol':>6}  {'ΔFert':>7}  {'ΔLut':>6}")
    print(f"  {'─'*36}  {'─'*9}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}")

    for max_w, ramp, label in configs:
        _set_weight_config(max_w, ramp)
        res = evaluate(df, use_oracle=use_oracle)
        overall, nm, f1 = _sweep_summary(
            res["treatment"]["truth"], res["treatment"]["pred"]
        )
        print(f"  {label:<36}  {overall*100:>8.2f}%  "
              f"{(overall - base_overall)*100:>+5.2f}  "
              f"{(f1['Follicular'] - base_f1['Follicular'])*100:>+5.2f}  "
              f"{(f1['Fertility']  - base_f1['Fertility']) *100:>+6.2f}  "
              f"{(f1['Luteal']     - base_f1['Luteal'])    *100:>+5.2f}")


def run_history_length_sweep(df: pd.DataFrame, use_oracle: bool) -> None:
    """
    Simulate users with increasingly long history by replicating each
    participant's own labeled data backwards in time before the evaluation
    window. Answers: does personalization actually matter when users have
    been logging for 6+ months?
    """
    # Each config: (pre_history_cycles, label)
    # 0 copies = real 3mo data only
    # 1 copy  = simulated 6mo history (1 extra copy of their data placed before day 1)
    # 2 copies = simulated 9mo
    # 3 copies = simulated 12mo
    configs = [
        (0, "3 months   (real data only)"),
        (1, "6 months   (1× synthetic prior)"),
        (2, "9 months   (2× synthetic prior)"),
        (3, "12 months  (3× synthetic prior)"),
    ]

    # Baseline (no personalization) — pre-history has no effect on baseline arm
    print("\n  Running baseline (no personalization)...")
    _set_weight_config(0.35, 30.0)   # use default weight
    base_res = evaluate(df, use_oracle=use_oracle, pre_history_cycles=0)
    base_o, base_nm, base_f1 = _sweep_summary(
        base_res["baseline"]["truth"], base_res["baseline"]["pred"]
    )
    print(f"\n  Baseline:  overall {base_o*100:.2f}%  non-men {base_nm*100:.2f}%  "
          f"F1s: Fol {base_f1['Follicular']*100:.1f} / "
          f"Fert {base_f1['Fertility']*100:.1f} / "
          f"Lut {base_f1['Luteal']*100:.1f}")

    print(f"\n  {'─'*76}")
    print(f"  {'Simulated history':<36}  {'Overall':>9}  {'ΔOvr':>6}  {'ΔFol':>6}  {'ΔFert':>7}  {'ΔLut':>6}")
    print(f"  {'─'*36}  {'─'*9}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}")

    for n_cycles, label in configs:
        res = evaluate(df, use_oracle=use_oracle, pre_history_cycles=n_cycles)
        overall, nm, f1 = _sweep_summary(
            res["treatment"]["truth"], res["treatment"]["pred"]
        )
        print(f"  {label:<36}  {overall*100:>8.2f}%  "
              f"{(overall - base_o)*100:>+5.2f}  "
              f"{(f1['Follicular'] - base_f1['Follicular'])*100:>+5.2f}  "
              f"{(f1['Fertility']  - base_f1['Fertility']) *100:>+6.2f}  "
              f"{(f1['Luteal']     - base_f1['Luteal'])    *100:>+5.2f}")


def main():
    parser = argparse.ArgumentParser(description="Progressive personalization evaluation")
    parser.add_argument("--all",      action="store_true",
                        help="Evaluate on all participants (default: held-out only)")
    parser.add_argument("--oracle",   action="store_true",
                        help="Seed history with ground-truth phase labels "
                             "(upper-bound test — shows what personalization "
                             "can achieve with perfect self-supervision).")
    parser.add_argument("--full-wave", action="store_true",
                        help="Include 2024 wave (default: 2022 only)")
    parser.add_argument("--sweep",     action="store_true",
                        help="Sweep blend-weight configurations to find a sweet spot")
    parser.add_argument("--history-sweep", action="store_true",
                        help="Simulate 3/6/9/12-month users by pre-seeding history "
                             "with copies of each participant's own labeled data.")
    parser.add_argument("--pre-history", type=int, default=0,
                        help="Pre-seed N copies of each participant's labeled data "
                             "before the evaluation window (simulates long-time users)")
    args = parser.parse_args()

    print(f"\nPersonalization — Progressive Held-Out Evaluation\n{'─'*65}")
    print(f"  Dataset: {CSV_PATH.name}")
    print(f"  Mode:    {'ORACLE (truth-labelled history)' if args.oracle else 'PRODUCTION (predicted history)'}")

    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded:  {len(df):,} rows | {df['id'].nunique()} participants")

    held_out_pids: set = set()
    if HELD_OUT_FILE.exists():
        import json
        with open(HELD_OUT_FILE) as f:
            held_out_pids = set(json.load(f).get("held_out_participant_ids", []))
        print(f"  Held-out participants: {sorted(held_out_pids)}")

    if not args.full_wave:
        df = df[df["study_interval"] == 2022]
        print(f"  Using 2022 wave: {len(df):,} rows")

    if args.all:
        print(f"  Evaluating ALL participants: {len(df):,} rows")
    elif held_out_pids:
        df = df[df["id"].isin(held_out_pids)]
        print(f"  Held-out only: {len(df):,} rows | {df['id'].nunique()} participants")
    else:
        print("  WARNING: no held_out_participants.json found — using all participants")

    print()
    if args.sweep:
        run_sweep(df, use_oracle=args.oracle)
        print()
        return

    if args.history_sweep:
        run_history_length_sweep(df, use_oracle=args.oracle)
        print()
        return

    results = evaluate(df, use_oracle=args.oracle,
                       pre_history_cycles=args.pre_history)
    if args.pre_history:
        print(f"  [Pre-seeded {args.pre_history}× participant-data copies "
              f"as synthetic prior history]")
    compute_comparison(results)
    per_participant_breakdown(results)
    print()


if __name__ == "__main__":
    main()
