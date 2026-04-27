#!/usr/bin/env python3
"""
Train Layer 2 v5 cycle phase model — trajectory-aware, 5-day history.

What v5 adds over v4
─────────────────────────────────────────────────────────────────────────────
1.  Ordinal severity trajectory features.
    v4 captures today's ordinal severity but has no lag or delta on the
    ordinal axis.  v5 adds {sym}_ord_lag1/lag2, group-level *_ord_delta,
    and total_severity_delta so the model can distinguish rising vs falling
    symptom loads — a key signal for the Follicular→Fertility and
    Fertility→Luteal transitions.

2.  Extended 5-day rolling window.
    v4 uses recent_daily_logs[-2:] (2 prior days).  v5 uses [-4:], giving
    up to 5 total days (4 prior + today) for roll5_mean/max features.

3.  Symptom pattern indicators.
    is_symptom_rising / declining / peak / trough — derived from the
    3-day ordinal trend.

4.  Flow score trajectory.
    Numeric flow intensity (0–5) with lag1/lag2 and flow_declining flag.

5.  Cycle interaction feature.
    cycle_cv_x_day_norm = cycle_length_cv × cycle_day_norm — captures
    "uncertain timing, late-cycle" cases that confuse the Luteal detector.

All mucus-derived features remain excluded (same leakage guard as v4).
Held-out participants are excluded from all training and CV (same as v4).

Run:  python scripts/train_layer2_v5.py
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cycle.layer1_period import get_layer1_output
from cycle.layer2.features_v3 import (
    SUPPORTED_SYMPTOMS,
    severities_to_binary_symptoms,
    to_ordinal,
)
from cycle.layer2.features_v5 import build_v5_feature_row
from cycle.config import ARTIFACTS_DIR

V5_PIPELINE_FILE      = "layer2_v5_pipeline.joblib"
V5_LABEL_ENCODER_FILE = "layer2_v5_label_encoder.joblib"
V5_FEATURE_COLS_FILE  = "layer2_v5_feature_columns.joblib"
V5_METADATA_FILE      = "layer2_v5_metadata.joblib"

HELD_OUT_FILE = ROOT / "scripts" / "held_out_participants.json"

# Mucus columns to drop — identical list to v4 (label-leakage guard)
MUCUS_COLUMNS_TO_DROP = {
    "mucus_fertility_score",
    "mucus_fertility_score_lag1",
    "mucus_fertility_score_lag2",
    "mucus_fertility_score_roll3_mean",
    "mucus_fertility_score_roll3_max",
    "mucus_fertility_score_trend2",
    "mucus_fertility_score_persist3",
    "mucus_logged",
    "mucus_score_logged",
    "mucus_type",
}


# ── CSV helpers ───────────────────────────────────────────────────────────────

def row_to_severities(row) -> dict:
    return {sym: to_ordinal(row.get(sym)) for sym in SUPPORTED_SYMPTOMS}


def row_to_flow(row):
    val = row.get("flow_volume")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return str(val).strip()


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(df: pd.DataFrame, held_out_pids: set):
    """
    Replay each 2022 participant's days in order, collecting v5 feature rows.

    Key differences from the v4 builder:
    - history_buffer retains up to 4 prior days (was 2)
    - Each log entry in recent_daily_logs includes ``symptom_severities``
      so the v5 feature builder can compute ordinal lags and deltas.
    """
    records = []
    skipped = 0
    BASE_DATE = datetime(2022, 1, 1)

    df_2022 = df[(df["study_interval"] == 2022) & (~df["id"].isin(held_out_pids))]
    print(f"  Training pool: {len(df_2022)} rows | "
          f"{df_2022['id'].nunique()} participants (2022, held-out excluded)")

    for pid, grp in df_2022.groupby("id", sort=True):
        grp = grp.sort_values("day_in_study").reset_index(drop=True)
        grp["date"] = grp["day_in_study"].apply(
            lambda d: (BASE_DATE + timedelta(days=int(d) - 1)).strftime("%Y-%m-%d")
        )

        known_starts   = []
        prev_men       = False
        history_buffer = []   # each entry: {severities, appetite, exerciselevel, flow}

        for _, row in grp.iterrows():
            phase_truth = str(row["phase"])
            today       = str(row["date"])
            is_men      = (phase_truth == "Menstrual")
            is_day1     = is_men and not prev_men

            if is_day1:
                known_starts.append(today)

            if not known_starts:
                skipped += 1
                prev_men = is_men
                continue

            severities    = row_to_severities(row)
            appetite      = to_ordinal(row.get("appetite"))
            exerciselevel = to_ordinal(row.get("exerciselevel"))
            flow          = row_to_flow(row)

            # v5: 4 prior days, including ordinal severities in each log
            recent_daily_logs = [
                {
                    "symptoms":           severities_to_binary_symptoms(h["severities"]),
                    "symptom_severities": h["severities"],   # v5 addition
                    "cervical_mucus":     "unknown",
                    "appetite":           h["appetite"],
                    "exerciselevel":      h["exerciselevel"],
                    "flow":               h["flow"],
                }
                for h in history_buffer[-4:]
            ]

            layer1 = get_layer1_output(list(known_starts), today=today)

            if phase_truth in ("nan", "None", "") or pd.isna(phase_truth):
                history_buffer.append({"severities": severities, "appetite": appetite,
                                        "exerciselevel": exerciselevel, "flow": flow})
                prev_men = is_men
                continue

            # Skip Menstrual rows — handled by fusion rules
            if is_day1 or (layer1.get("cycle_day") and layer1["cycle_day"] <= 5) or is_men:
                history_buffer.append({"severities": severities, "appetite": appetite,
                                        "exerciselevel": exerciselevel, "flow": flow})
                prev_men = is_men
                continue

            feat = build_v5_feature_row(
                symptom_severities=severities,
                cervical_mucus="unknown",
                appetite=appetite,
                exerciselevel=exerciselevel,
                flow=flow,
                recent_daily_logs=recent_daily_logs,
                layer1=layer1,
                known_starts=list(known_starts),
            )
            feat["_label"]       = phase_truth
            feat["_participant"] = pid
            records.append(feat)

            history_buffer.append({"severities": severities, "appetite": appetite,
                                    "exerciselevel": exerciselevel, "flow": flow})
            prev_men = is_men

    print(f"  Skipped {skipped} rows (no prior period history)")
    print(f"  Feature rows collected: {len(records)}")

    data_df    = pd.DataFrame(records)
    y          = data_df.pop("_label")
    groups_pid = data_df.pop("_participant")
    return data_df, y, groups_pid


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_pipeline(numeric_features, categorical_features):
    num_pipe = SKPipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = SKPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ])
    clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=15,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return SKPipeline([("preprocessor", preprocessor), ("model", clf)])


def print_results(y_true, y_pred, title: str):
    phases = ["Follicular", "Fertility", "Luteal"]
    sep = "=" * 62
    print(f"\n{sep}\n  {title}\n{sep}")
    total   = len(y_true)
    correct = (y_true == y_pred).sum()
    print(f"  Accuracy:           {correct}/{total} = {correct/total:.1%}")
    print(f"  Balanced accuracy:  {balanced_accuracy_score(y_true, y_pred):.1%}")
    print(f"  Macro F1:           {f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}")
    print("\n  Per-phase recall:")
    for p in phases:
        mask = y_true == p
        if mask.sum() == 0:
            continue
        rec = (y_pred[mask] == p).mean()
        print(f"    {p:12s}  n={mask.sum():4d}  recall={rec:.1%}")
    print("\n  Confusion matrix (rows=truth, cols=pred):")
    print(pd.crosstab(y_true, y_pred).to_string())
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    if HELD_OUT_FILE.exists():
        with open(HELD_OUT_FILE) as f:
            held_out_pids = set(json.load(f)["held_out_participant_ids"])
        print(f"Held-out participants (excluded from all training): {sorted(held_out_pids)}")
    else:
        print("WARNING: held_out_participants.json not found — training on all participants.")
        held_out_pids = set()

    csv_path = ROOT / "data" / "mcphases_with_estimated_cervical_mucus_v3.csv"
    print(f"\nLoading {csv_path.name} …")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows | {df['id'].nunique()} participants total")
    print(f"  Using 2022 only (2024 has 99.4% symptom missingness)\n")

    print("Building feature matrix (v5 — trajectory + 5-day window, no mucus) …")
    X, y, groups_pid = build_dataset(df, held_out_pids)

    # ── Drop mucus columns ────────────────────────────────────────────────────
    mucus_cols_present = [c for c in MUCUS_COLUMNS_TO_DROP if c in X.columns]
    X = X.drop(columns=mucus_cols_present)
    print(f"\n  Dropped {len(mucus_cols_present)} mucus-derived columns (label leakage)")

    # ── Feature classification ────────────────────────────────────────────────
    categorical_features = [c for c in ["exerciselevel", "appetite"] if c in X.columns]
    numeric_features     = [c for c in X.columns if c not in categorical_features]
    all_feature_cols     = numeric_features + categorical_features

    X = X[all_feature_cols]
    print(f"  Total features (mucus-free): {len(all_feature_cols)}")
    print(f"  v5 new features: {len(all_feature_cols)} total  "
          f"(v4 had ~{len(all_feature_cols) - 40} fewer)")
    print(f"  Full dataset for CV: {len(X):,} rows  {dict(y.value_counts())}")

    # ── LOPO CV ───────────────────────────────────────────────────────────────
    print(f"\nRunning Leave-One-Participant-Out CV ({groups_pid.nunique()} folds) …")
    logo = LeaveOneGroupOut()
    oof_true, oof_pred = [], []
    fold_results = []

    for fold, (tr, va) in enumerate(logo.split(X, y, groups=groups_pid), 1):
        pid_val = groups_pid.iloc[va[0]]
        pipe = build_pipeline(numeric_features, categorical_features)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        preds = pipe.predict(X.iloc[va])

        fold_acc = (y.iloc[va].values == preds).mean()
        fold_bal = balanced_accuracy_score(y.iloc[va], preds)
        fold_results.append({"pid": pid_val, "n": len(va), "acc": fold_acc, "bal_acc": fold_bal})
        oof_true.extend(y.iloc[va].tolist())
        oof_pred.extend(preds.tolist())

        if fold % 5 == 0 or fold == groups_pid.nunique():
            print(f"  Fold {fold:2d}/{groups_pid.nunique()} (pid={pid_val:3d}, n={len(va):3d})  "
                  f"acc={fold_acc:.1%}  bal_acc={fold_bal:.1%}")

    oof_true = pd.Series(oof_true)
    oof_pred = pd.Series(oof_pred)
    print_results(oof_true, oof_pred, "LOPO CV — participant-level out-of-sample (HONEST ESTIMATE)")

    fold_df = pd.DataFrame(fold_results).sort_values("bal_acc")
    print("  Lowest 5 participants by balanced accuracy:")
    for _, r in fold_df.head(5).iterrows():
        print(f"    pid={int(r['pid']):3d}  n={int(r['n']):3d}  bal_acc={r['bal_acc']:.1%}")

    # ── Final fit on all eligible 2022 data ───────────────────────────────────
    print("\nTraining final model on full 2022 eligible dataset …")
    label_encoder = LabelEncoder().fit(y)
    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X, y)
    print("  Done.")

    y_pred_full = pipeline.predict(X)
    print_results(y, pd.Series(y_pred_full, index=y.index),
                  "FULL TRAINING SET — in-sample sanity (NOT a generalisation metric)")

    # ── Save artifacts ────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline,         ARTIFACTS_DIR / V5_PIPELINE_FILE)
    joblib.dump(label_encoder,    ARTIFACTS_DIR / V5_LABEL_ENCODER_FILE)
    joblib.dump(all_feature_cols, ARTIFACTS_DIR / V5_FEATURE_COLS_FILE)
    joblib.dump({
        "numeric_features":         numeric_features,
        "categorical_features":     categorical_features,
        "mucus_features_excluded":  sorted(mucus_cols_present),
        "trained_on":               "2022_only",
        "held_out_pids":            sorted(held_out_pids),
        "cv_method":                "LOPO_by_participant",
        "best_model_name":          "lgbm_600_v5_trajectory",
        "lopo_oof_balanced_acc":    float(balanced_accuracy_score(oof_true, oof_pred)),
        "lopo_oof_macro_f1":        float(f1_score(oof_true, oof_pred, average="macro", zero_division=0)),
        "new_feature_groups": [
            "group_ord_delta (pain/energy/mood)",
            "total_severity_delta",
            "extended_roll5 (total/pain)",
            "symptom_pattern_indicators (rising/declining/peak/trough)",
            "flow_score_trajectory (score/lag1/declining)",
            "cycle_cv_x_day_norm",
        ],
        "note": "Honest metric = lopo_oof_balanced_acc. In-sample number is inflated.",
    }, ARTIFACTS_DIR / V5_METADATA_FILE)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    for f in [V5_PIPELINE_FILE, V5_LABEL_ENCODER_FILE, V5_FEATURE_COLS_FILE, V5_METADATA_FILE]:
        print(f"  {f}")

    lopo_bal  = balanced_accuracy_score(oof_true, oof_pred)
    lopo_mac  = f1_score(oof_true, oof_pred, average="macro", zero_division=0)
    print(f"\n{'='*62}")
    print(f"  HONEST GENERALISATION ESTIMATE (LOPO CV)")
    print(f"  Balanced accuracy: {lopo_bal:.1%}")
    print(f"  Macro F1:          {lopo_mac:.3f}")
    print(f"{'='*62}")
    print(f"\nThis is the number to report. NOT the in-sample sanity figure.")
    print(f"To evaluate on held-out participants:")
    print(f"  INBALANCE_MODEL_VERSION=v5 python scripts/evaluate_cycle.py --held-out")


if __name__ == "__main__":
    main()
