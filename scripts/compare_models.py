#!/usr/bin/env python3
"""
Full KPI comparison: v1 (current) vs v2 vs v3 models through the fusion pipeline.

Each version runs through its own correct path:
  • v1 — binary symptoms, legacy cycle_day<=5 menstrual-continuation, v1 fusion weights
  • v2 — binary symptoms + 2 cycle features, legacy menstrual rule, v1 fusion weights
  • v3 — ordinal severity features (build_v3_feature_row), flow-aware menstrual rule
         (_is_ongoing_menstrual), v3-tuned fusion weights

KPIs reported
─────────────
  Accuracy            — standard % correct
  Balanced accuracy   — mean recall across classes (handles imbalance)
  Macro F1            — unweighted mean F1 across all phases
  Cohen's Kappa       — agreement adjusted for chance
  Phase MAE (steps)   — mean |predicted_phase_ordinal − true_phase_ordinal|
  Cycle-day MAE       — mean |predicted_midday − true_midday|
                        Menstrual=3, Follicular=8, Fertility=14, Luteal=22
  Adjacent-error rate — % of wrong predictions off by only 1 phase
  Per-phase P/R/F1

Usage:  python tests/compare_models.py
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)

from cycle.layer1_period import (
    compute_cycle_lengths,
    get_layer1_output,
)
from cycle.layer2.base import (
    _build_recent_rows,
    _build_today_row,
    _apply_history_features,
    _get_signal_confidence,
    _normalize_symptom_list,
    MUCUS_FERTILITY_MAP,
    _normalize_mucus,
    _make_feature_frame,
)
from cycle.fusion import (
    has_symptom_input,
    _map_layer1_to_non_menstrual,
    _get_layer1_non_menstrual_top_phase,
    _fuse_non_menstrual_probs,
    _constrain_non_menstrual_probs,
    _is_ongoing_menstrual,
)
from cycle.config import NON_MENSTRUAL_PHASES, ARTIFACTS_DIR, SUPPORTED_SYMPTOMS
from cycle.layer2.features_v3 import (
    build_v3_feature_row,
    severities_to_binary_symptoms,
    to_ordinal,
)

# ── Phase → cycle day midpoint ────────────────────────────────────────────────
PHASE_DAY = {"Menstrual": 3, "Follicular": 8, "Fertility": 14, "Luteal": 22}
PHASE_ORD = {"Menstrual": 0, "Follicular": 1, "Fertility": 2, "Luteal": 3}

# ── Ordinal / string helpers ──────────────────────────────────────────────────
_ORDINAL_INT = {
    "Not at all":      0,
    "Very Low":        1,
    "Very Low/Little": 1,
    "Low":             2,
    "Moderate":        3,
    "High":            4,
    "Very High":       5,
}
SYMPTOM_THRESHOLD = 3
SYMPTOM_COLS = list(SUPPORTED_SYMPTOMS)
MUCUS_MAP = {"egg_white": "eggwhite"}


def _to_int(val) -> int:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    s = str(val).strip()
    if s in _ORDINAL_INT:
        return _ORDINAL_INT[s]
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def row_to_symptoms(row) -> list:
    return [col for col in SYMPTOM_COLS if _to_int(row.get(col)) >= SYMPTOM_THRESHOLD]


def row_to_severities(row) -> dict:
    return {sym: to_ordinal(row.get(sym)) for sym in SUPPORTED_SYMPTOMS}


def row_to_mucus(row) -> str:
    val = row.get("cervical_mucus_estimated_type_v3")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "unknown"
    s = str(val).strip().lower()
    s = MUCUS_MAP.get(s, s)
    return s if s in {"dry", "sticky", "creamy", "watery", "eggwhite"} else "unknown"


def row_to_flow_legacy(row) -> str:
    """Legacy v1/v2 path: NaN collapses to 'none'."""
    val = row.get("flow_volume", "none")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "none"
    return str(val).strip()


def row_to_flow_v3(row):
    """v3 path: NaN is kept as None so _flow_category can return 'missing'."""
    val = row.get("flow_volume")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return str(val).strip()


# ── Feature frame builders ────────────────────────────────────────────────────

def make_frame_v1(symptoms, mucus, appetite, exerciselevel, flow, recent_daily_logs, layer1):
    """Original 116-feature frame (v1 model)."""
    return _make_feature_frame(symptoms, mucus, appetite, exerciselevel, recent_daily_logs, flow)


def make_frame_v2(symptoms, mucus, appetite, exerciselevel, flow, recent_daily_logs, layer1):
    """116 + cycle_day_norm + days_from_ovulation (v2 model)."""
    today_row    = _build_today_row(symptoms, mucus, appetite, exerciselevel, flow)
    history_rows = _build_recent_rows(recent_daily_logs, today_row)
    final_row    = _apply_history_features(history_rows, today_row)

    cycle_day = layer1.get("cycle_day")
    avg_len   = layer1.get("estimated_cycle_length")
    ovul_day  = layer1.get("possible_ovulation_day")

    if cycle_day is not None and avg_len and avg_len > 0:
        final_row["cycle_day_norm"]      = cycle_day / avg_len
        final_row["days_from_ovulation"] = float(cycle_day - ovul_day) if ovul_day else 0.0
    else:
        final_row["cycle_day_norm"]      = 0.5
        final_row["days_from_ovulation"] = 0.0

    feat_cols = joblib.load(ARTIFACTS_DIR / "layer2_v2_feature_columns.joblib")
    X = pd.DataFrame([{c: final_row.get(c, np.nan) for c in feat_cols}])
    for c in X.columns:
        if X[c].dtype == "object" and X[c].isna().all():
            X[c] = "unknown"
    return X


def make_frame_v3(severities, mucus, appetite_ord, exerciselevel_ord, flow,
                  recent_daily_logs, layer1, known_starts):
    """v3 ordinal-severity feature builder."""
    final_row = build_v3_feature_row(
        symptom_severities=severities,
        cervical_mucus=mucus,
        appetite=appetite_ord,
        exerciselevel=exerciselevel_ord,
        flow=flow,
        recent_daily_logs=recent_daily_logs,
        layer1=layer1,
        known_starts=known_starts,
    )
    feat_cols = joblib.load(ARTIFACTS_DIR / "layer2_v3_feature_columns.joblib")
    X = pd.DataFrame([{c: final_row.get(c, np.nan) for c in feat_cols}])
    for c in X.columns:
        if X[c].dtype == "object" and X[c].isna().all():
            X[c] = "unknown"
    return X


# ── Fusion helpers ────────────────────────────────────────────────────────────

def _apply_fusion(layer1, phase_probs, symptoms, mucus, severities=None,
                  model_version: str = "v1") -> str:
    layer1_probs_3 = _map_layer1_to_non_menstrual(layer1["phase_probs"])
    baseline       = _get_layer1_non_menstrual_top_phase(layer1_probs_3)
    mucus_score    = MUCUS_FERTILITY_MAP.get(_normalize_mucus(mucus), 0.0)
    fused          = _fuse_non_menstrual_probs(
        layer1_probs_3, phase_probs,
        mucus_fertility_score=mucus_score,
        model_version=model_version,
    )

    sorted_items   = sorted(phase_probs.items(), key=lambda x: x[1], reverse=True)
    top_prob       = sorted_items[0][1]
    second_prob    = sorted_items[1][1] if len(sorted_items) > 1 else 0.0

    if severities is not None:
        sym_count = sum(1 for v in severities.values() if v >= 3)
    else:
        sym_count = len(_normalize_symptom_list(symptoms))

    layer2_meta = {
        "top_phase":         sorted_items[0][0],
        "top_prob":          top_prob,
        "second_prob":       second_prob,
        "prob_gap":          top_prob - second_prob,
        "signal_confidence": _get_signal_confidence(phase_probs, sym_count, mucus),
    }
    constrained = _constrain_non_menstrual_probs(
        fused, baseline, layer2_meta, model_version=model_version,
    )
    return max(constrained, key=constrained.get)


# ── Per-model batch predict ───────────────────────────────────────────────────

def batch_predict(pipeline, class_names, frames):
    if not frames:
        return []
    batch_df  = pd.concat(frames, ignore_index=True)
    batch_pr  = pipeline.predict_proba(batch_df)
    results   = []
    for probs_row in batch_pr:
        phase_probs = {phase: float(p) for phase, p in zip(class_names, probs_row)}
        for p in NON_MENSTRUAL_PHASES:
            phase_probs.setdefault(p, 0.0)
        total = sum(phase_probs.values()) or 1.0
        results.append({k: v / total for k, v in phase_probs.items()})
    return results


# ── KPI calculation ───────────────────────────────────────────────────────────

def compute_kpis(res: pd.DataFrame) -> dict:
    valid = res.dropna(subset=["truth", "predicted"])
    valid = valid[valid["truth"].isin(PHASE_DAY) & valid["predicted"].isin(PHASE_DAY)]

    y_true = valid["truth"]
    y_pred = valid["predicted"]

    ord_true = y_true.map(PHASE_ORD)
    ord_pred = y_pred.map(PHASE_ORD)
    ord_mae  = (ord_true - ord_pred).abs().mean()

    day_true = y_true.map(PHASE_DAY)
    day_pred = y_pred.map(PHASE_DAY)
    day_mae  = (day_true - day_pred).abs().mean()

    wrong      = valid[y_true != y_pred]
    adj_errors = (wrong["truth"].map(PHASE_ORD) - wrong["predicted"].map(PHASE_ORD)).abs() == 1
    adj_rate   = adj_errors.mean() if len(wrong) > 0 else 0.0

    all_phases = ["Menstrual", "Follicular", "Fertility", "Luteal"]
    present    = [p for p in all_phases if p in y_true.values]

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average=None, zero_division=0
    )

    return {
        "accuracy":          (y_true == y_pred).mean(),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1":          f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa":             cohen_kappa_score(y_true, y_pred),
        "phase_mae":         ord_mae,
        "day_mae":           day_mae,
        "adj_error_rate":    adj_rate,
        "per_phase": {
            p: {"precision": prec[i], "recall": rec[i], "f1": f1[i]}
            for i, p in enumerate(present)
        },
    }


# ── Main comparison ───────────────────────────────────────────────────────────

def run():
    csv_path = os.path.join(ROOT, "data", "mcphases_with_estimated_cervical_mucus_v3.csv")
    df = pd.read_csv(csv_path)

    print("Loading models …")
    models = {}
    for ver, pfile, lfile in [
        ("v1", "layer2_vnext_pipeline.joblib",      "layer2_vnext_label_encoder.joblib"),
        ("v2", "layer2_v2_pipeline.joblib",         "layer2_v2_label_encoder.joblib"),
        ("v3", "layer2_v3_pipeline.joblib",         "layer2_v3_label_encoder.joblib"),
    ]:
        pipe = joblib.load(ARTIFACTS_DIR / pfile)
        try:
            le     = joblib.load(ARTIFACTS_DIR / lfile)
            cnames = le.inverse_transform(pipe.classes_)
        except Exception:
            cnames = pipe.classes_   # v2/v3 store string labels directly
        models[ver] = (pipe, cnames)
        print(f"  {ver} loaded — classes: {list(cnames)}")
    print()

    BASE_DATES = {2022: datetime(2022, 1, 1), 2024: datetime(2024, 1, 1)}

    meta_rows     = []
    frames        = {"v1": [], "v2": [], "v3": []}
    # Each model may skip a row (menstrual override) independently, so we track
    # separate lists of which meta-row indices need inference per model.
    frame_indices = {"v1": [], "v2": [], "v3": []}
    skipped = 0

    for (pid, year), grp in df.groupby(["id", "study_interval"], sort=True):
        grp = grp.sort_values("day_in_study").reset_index(drop=True)
        base = BASE_DATES.get(int(year), datetime(2022, 1, 1))
        grp["date"] = grp["day_in_study"].apply(
            lambda d: (base + timedelta(days=int(d) - 1)).strftime("%Y-%m-%d")
        )

        known_starts   = []
        prev_men       = False
        # Two history buffers — the legacy one stores binary symptoms; the v3
        # one stores ordinal severities plus preserved flow/None.
        history_buffer_legacy = []
        history_buffer_v3     = []

        for _, row in grp.iterrows():
            phase_truth    = str(row["phase"])
            today          = str(row["date"])
            is_men         = (phase_truth == "Menstrual")
            is_period_day1 = is_men and not prev_men

            if is_period_day1:
                known_starts.append(today)
            if not known_starts:
                skipped += 1
                prev_men = is_men
                continue

            symptoms      = row_to_symptoms(row)
            severities    = row_to_severities(row)
            mucus         = row_to_mucus(row)
            appetite      = _to_int(row.get("appetite"))
            exerciselevel = _to_int(row.get("exerciselevel"))
            appetite_ord      = to_ordinal(row.get("appetite"))
            exerciselevel_ord = to_ordinal(row.get("exerciselevel"))
            flow_legacy   = row_to_flow_legacy(row)
            flow_v3       = row_to_flow_v3(row)

            recent_daily_logs_legacy = [
                {"symptoms": h["symptoms"], "cervical_mucus": h["mucus"],
                 "appetite": h["appetite"], "exerciselevel": h["exerciselevel"],
                 "flow": h["flow"]}
                for h in history_buffer_legacy[-2:]
            ]
            recent_daily_logs_v3 = [
                {
                    "symptoms":       severities_to_binary_symptoms(h["severities"]),
                    "cervical_mucus": h["mucus"],
                    "appetite":       h["appetite"],
                    "exerciselevel":  h["exerciselevel"],
                    "flow":           h["flow"],
                }
                for h in history_buffer_v3[-2:]
            ]

            layer1 = get_layer1_output(list(known_starts), today=today)

            # Per-model menstrual-continuation decisions
            legacy_men_ongoing = (
                (not is_period_day1)
                and layer1.get("cycle_day") is not None
                and layer1["cycle_day"] <= 5
            )
            v3_men_ongoing = (
                (not is_period_day1)
                and _is_ongoing_menstrual(layer1, flow_v3, recent_daily_logs_v3)
            )

            needs_l2_legacy = (not is_period_day1) and (not legacy_men_ongoing) and has_symptom_input(
                symptoms=symptoms, cervical_mucus=mucus,
                appetite=appetite, exerciselevel=exerciselevel,
                recent_daily_logs=recent_daily_logs_legacy,
            )
            needs_l2_v3 = (not is_period_day1) and (not v3_men_ongoing) and has_symptom_input(
                symptoms=severities_to_binary_symptoms(severities),
                cervical_mucus=mucus,
                appetite=appetite_ord,
                exerciselevel=exerciselevel_ord,
                recent_daily_logs=recent_daily_logs_v3,
                symptom_severities=severities,
            )

            meta = {
                "truth":          phase_truth,
                "is_period_day1": is_period_day1,
                "symptoms":       symptoms,
                "severities":     severities,
                "mucus":          mucus,
                "layer1":         layer1,
                "predicted_v1":   None,
                "predicted_v2":   None,
                "predicted_v3":   None,
                "mode":           None,
            }
            idx_now = len(meta_rows)

            # v1 / v2 path (legacy menstrual rule)
            if is_period_day1:
                meta["predicted_v1"] = meta["predicted_v2"] = "Menstrual"
            elif legacy_men_ongoing:
                meta["predicted_v1"] = meta["predicted_v2"] = "Menstrual"
            elif not needs_l2_legacy:
                l1p3 = _map_layer1_to_non_menstrual(layer1["phase_probs"])
                pred = _get_layer1_non_menstrual_top_phase(l1p3)
                meta["predicted_v1"] = meta["predicted_v2"] = pred
            else:
                frame_indices["v1"].append(idx_now)
                frame_indices["v2"].append(idx_now)
                frames["v1"].append(make_frame_v1(
                    symptoms, mucus, appetite, exerciselevel, flow_legacy,
                    recent_daily_logs_legacy, layer1))
                frames["v2"].append(make_frame_v2(
                    symptoms, mucus, appetite, exerciselevel, flow_legacy,
                    recent_daily_logs_legacy, layer1))

            # v3 path (flow-aware menstrual rule, ordinal features)
            if is_period_day1:
                meta["predicted_v3"] = "Menstrual"
            elif v3_men_ongoing:
                meta["predicted_v3"] = "Menstrual"
            elif not needs_l2_v3:
                l1p3 = _map_layer1_to_non_menstrual(layer1["phase_probs"])
                meta["predicted_v3"] = _get_layer1_non_menstrual_top_phase(l1p3)
            else:
                frame_indices["v3"].append(idx_now)
                frames["v3"].append(make_frame_v3(
                    severities, mucus, appetite_ord, exerciselevel_ord, flow_v3,
                    recent_daily_logs_v3, layer1, list(known_starts)))

            # Shared mode label (uses v3's decision for display)
            if is_period_day1:
                meta["mode"] = "period_start_override"
            elif v3_men_ongoing or legacy_men_ongoing:
                meta["mode"] = "menstrual_ongoing"
            elif not needs_l2_v3:
                meta["mode"] = "layer1_only_non_menstrual"
            else:
                meta["mode"] = "fused_non_menstrual"

            meta_rows.append(meta)
            prev_men = is_men
            history_buffer_legacy.append({
                "symptoms": symptoms, "mucus": mucus,
                "appetite": appetite, "exerciselevel": exerciselevel, "flow": flow_legacy,
            })
            history_buffer_v3.append({
                "severities": severities, "mucus": mucus,
                "appetite": appetite_ord, "exerciselevel": exerciselevel_ord,
                "flow": flow_v3,
            })

    print(f"Skipped {skipped} rows.")
    print(f"L2 rows — v1: {len(frame_indices['v1']):,}  v2: {len(frame_indices['v2']):,}  "
          f"v3: {len(frame_indices['v3']):,}\n")
    print("Running batch predictions for all three models …")

    for ver in ["v1", "v2", "v3"]:
        pipe, cnames = models[ver]
        probs_list = batch_predict(pipe, cnames, frames[ver])
        for row_idx, phase_probs in zip(frame_indices[ver], probs_list):
            meta = meta_rows[row_idx]
            if ver == "v3":
                meta["predicted_v3"] = _apply_fusion(
                    meta["layer1"], phase_probs, meta["symptoms"], meta["mucus"],
                    severities=meta["severities"], model_version="v3",
                )
            else:
                meta[f"predicted_{ver}"] = _apply_fusion(
                    meta["layer1"], phase_probs, meta["symptoms"], meta["mucus"],
                    model_version="v1",
                )

    print("Done. Computing KPIs …\n")

    results = {}
    for ver in ["v1", "v2", "v3"]:
        records = [
            {
                "truth":          m["truth"],
                "predicted":      m[f"predicted_{ver}"],
                "is_period_day1": m["is_period_day1"],
                "mode":           m["mode"],
            }
            for m in meta_rows
        ]
        results[ver] = pd.DataFrame(records)

    _print_comparison(results)


def _print_comparison(results: dict):
    sep  = "=" * 70
    sep2 = "-" * 70

    kpis = {ver: compute_kpis(df) for ver, df in results.items()}

    print(sep)
    print("  MODEL COMPARISON — FULL PIPELINE (L1 + L2 + Fusion)")
    print(sep)
    print(f"  {'Metric':<28} {'v1 (current)':>14} {'v2 (+cycle day)':>16} {'v3 (LightGBM)':>14}")
    print(sep2)

    rows = [
        ("Accuracy",           "accuracy",          "{:.1%}"),
        ("Balanced accuracy",  "balanced_accuracy",  "{:.1%}"),
        ("Macro F1",           "macro_f1",           "{:.3f}"),
        ("Cohen's Kappa",      "kappa",              "{:.3f}"),
        ("Phase MAE (steps)",  "phase_mae",          "{:.3f}"),
        ("Cycle-day MAE (days)", "day_mae",          "{:.2f}"),
        ("Adjacent-error rate","adj_error_rate",     "{:.1%}"),
    ]

    for label, key, fmt in rows:
        vals = [fmt.format(kpis[v][key]) for v in ["v1", "v2", "v3"]]
        print(f"  {label:<28} {vals[0]:>14} {vals[1]:>16} {vals[2]:>14}")

    print(sep2)
    for ver in ["v1", "v2", "v3"]:
        res = results[ver]
        nm  = res[res["truth"] != "Menstrual"]
        men = res[res["truth"] == "Menstrual"]
        nm_acc  = (nm["truth"] == nm["predicted"]).mean() if len(nm) else 0.0
        men_acc = (men["truth"] == men["predicted"]).mean() if len(men) else 0.0
        print(f"  {ver}  Non-menstrual acc: {nm_acc:.1%}   Menstrual all-days: {men_acc:.1%}")

    print(f"\n{sep}")
    print("  PER-PHASE RECALL")
    print(sep)
    phases = ["Menstrual", "Follicular", "Fertility", "Luteal"]
    print(f"  {'Phase':<12} {'v1':>10} {'v2':>10} {'v3':>10}   Target")
    print(sep2)
    targets = {
        "Menstrual":   "≥70%",
        "Follicular":  "≥55%",
        "Fertility":   "≥40%",
        "Luteal":      "75–90%",
    }
    for p in phases:
        vals = []
        for ver in ["v1", "v2", "v3"]:
            res = results[ver]
            sub = res[res["truth"] == p]
            rec = (sub["truth"] == sub["predicted"]).mean() if len(sub) else 0.0
            vals.append(f"{rec:.1%}")
        print(f"  {p:<12} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}   {targets.get(p,'')}")

    print(f"\n{sep}")
    print("  PER-PHASE F1")
    print(sep)
    print(f"  {'Phase':<12} {'v1':>10} {'v2':>10} {'v3':>10}")
    print(sep2)
    for p in phases:
        vals = []
        for ver in ["v1", "v2", "v3"]:
            pp = kpis[ver]["per_phase"]
            f1 = pp.get(p, {}).get("f1", 0.0)
            vals.append(f"{f1:.3f}")
        print(f"  {p:<12} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    print(f"\n{sep}")
    print("  CYCLE-DAY MAE BREAKDOWN (mean |pred_midday − truth_midday| in days)")
    print(sep)
    print(f"  {'True phase':<12} {'v1':>10} {'v2':>10} {'v3':>10}   midday")
    print(sep2)
    for p in phases:
        vals = []
        for ver in ["v1", "v2", "v3"]:
            res = results[ver]
            sub = res[res["truth"] == p].dropna(subset=["predicted"])
            sub = sub[sub["predicted"].isin(PHASE_DAY)]
            mae = (sub["predicted"].map(PHASE_DAY) - PHASE_DAY[p]).abs().mean() if len(sub) else 0.0
            vals.append(f"{mae:.1f}")
        print(f"  {p:<12} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}   day {PHASE_DAY[p]}")

    print(f"\n{sep}")
    print("  ERROR SEVERITY — % of all errors that are N phases off")
    print(sep)
    print(f"  {'Off by':>8} {'v1':>10} {'v2':>10} {'v3':>10}")
    print(sep2)
    for dist in [1, 2, 3]:
        vals = []
        for ver in ["v1", "v2", "v3"]:
            res   = results[ver]
            valid = res[res["truth"].isin(PHASE_DAY) & res["predicted"].isin(PHASE_DAY)]
            wrong = valid[valid["truth"] != valid["predicted"]]
            if len(wrong) == 0:
                vals.append("—")
                continue
            d = (wrong["truth"].map(PHASE_ORD) - wrong["predicted"].map(PHASE_ORD)).abs()
            vals.append(f"{(d == dist).mean():.1%}")
        print(f"  {dist} phase(s)  {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")


if __name__ == "__main__":
    run()
