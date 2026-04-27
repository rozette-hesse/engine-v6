#!/usr/bin/env python3
"""
Train Layer 2 v4 cycle phase model — mucus-clean, LOPO CV.

Critical fixes over v3
───────────────────────────────────────────────────────────────────────────────
1.  ALL mucus features removed.
    cervical_mucus_estimated_type_v3 was derived from the phase label
    (source column: 'estimated_from_phase_hormones_flow_temperature').
    Feature-importance analysis on v3 showed the mucus_fertility_score family
    accounted for 69.6% of model gain — a textbook label-leakage signal.
    In production, users supply self-reported mucus, not label-derived estimates,
    so the training/deployment distributions would diverge. All mucus columns
    are excluded from training and inference.

2.  Training restricted to 2022 study wave only.
    The 2024 wave has 99.4% symptom missingness (only 12 of 1,961 rows have
    any symptom data). Including 2024 dilutes the feature matrix without adding
    reusable symptom-phase signal.

3.  Held-out participants excluded from all training and CV.
    Participants in held_out_participants.json are reserved for a single
    final evaluation. They must not appear in any training or CV fold.

4.  Leave-One-Participant-Out (LOPO) cross-validation.
    With 34 eligible participants the LOPO gives 34 folds and is the most
    conservative honest estimate for this dataset size — every participant's
    rows land in exactly one validation set.

5.  LightGBM with mild regularisation (same hypers as v3, grid-searchable).

Artifacts saved as layer2_v4_*.joblib — does NOT overwrite v3.
Run:  python train_layer2_v4.py
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

ROOT = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(ROOT))

from cycle.layer1_period import get_layer1_output
from cycle.layer2.features_v3 import (
    SUPPORTED_SYMPTOMS,
    build_v3_feature_row,
    severities_to_binary_symptoms,
    to_ordinal,
)
from cycle.config import ARTIFACTS_DIR

V4_PIPELINE_FILE      = "layer2_v4_pipeline.joblib"
V4_LABEL_ENCODER_FILE = "layer2_v4_label_encoder.joblib"
V4_FEATURE_COLS_FILE  = "layer2_v4_feature_columns.joblib"
V4_METADATA_FILE      = "layer2_v4_metadata.joblib"

HELD_OUT_FILE = ROOT / "scripts" / "held_out_participants.json"

MUCUS_MAP = {"egg_white": "eggwhite"}

# ── Mucus columns to drop (all derived from phase labels) ─────────────────────
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
    "mucus_type",          # categorical, but still label-derived — drop it
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
    Returns (X_df, y, groups_pid) for 2022-only, non-held-out rows.
    Mucus is passed as 'unknown' so the feature builder still runs
    (filling history lags correctly), but mucus columns are dropped later.
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

        known_starts = []
        prev_men = False
        history_buffer = []

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

            # Pass mucus as unknown — mucus columns are dropped after feature build
            recent_daily_logs = [
                {
                    "symptoms":       severities_to_binary_symptoms(h["severities"]),
                    "cervical_mucus": "unknown",
                    "appetite":       h["appetite"],
                    "exerciselevel":  h["exerciselevel"],
                    "flow":           h["flow"],
                }
                for h in history_buffer[-2:]
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

            feat = build_v3_feature_row(
                symptom_severities=severities,
                cervical_mucus="unknown",   # strip mucus signal at source
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

    data_df = pd.DataFrame(records)
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
    sep = "=" * 58
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

    # Load held-out participant list
    if HELD_OUT_FILE.exists():
        with open(HELD_OUT_FILE) as f:
            held_out_pids = set(json.load(f)["held_out_participant_ids"])
        print(f"Held-out participants (excluded from all training): {sorted(held_out_pids)}")
    else:
        print("WARNING: held_out_participants.json not found. Run held-out selection first.")
        held_out_pids = set()

    csv_path = ROOT / "data" / "mcphases_with_estimated_cervical_mucus_v3.csv"
    print(f"\nLoading {csv_path.name} …")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows | {df['id'].nunique()} participants total")
    print(f"  Using 2022 only (2024 has 99.4% symptom missingness)\n")

    print("Building feature matrix (v4 — no mucus leakage, ordinal severities) …")
    X, y, groups_pid = build_dataset(df, held_out_pids)

    # ── Drop mucus columns ────────────────────────────────────────────────────
    mucus_cols_present = [c for c in MUCUS_COLUMNS_TO_DROP if c in X.columns]
    X = X.drop(columns=mucus_cols_present)
    print(f"\n  Dropped {len(mucus_cols_present)} mucus-derived columns (label leakage):")
    for c in sorted(mucus_cols_present):
        print(f"    - {c}")

    # ── Feature classification ────────────────────────────────────────────────
    # mucus_type is categorical — exclude it too
    categorical_features = [c for c in ["exerciselevel", "appetite"] if c in X.columns]
    # mucus_type excluded intentionally
    numeric_features     = [c for c in X.columns if c not in categorical_features]
    all_feature_cols     = numeric_features + categorical_features

    X = X[all_feature_cols]
    print(f"\n  Total features (mucus-free): {len(all_feature_cols)}")
    print(f"  Numeric: {len(numeric_features)}   Categorical: {len(categorical_features)}")
    print(f"  Full dataset for CV: {len(X):,} rows  {dict(y.value_counts())}")

    # ── LOPO CV — honest out-of-sample estimate ───────────────────────────────
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

    # Worst-performing participants
    fold_df = pd.DataFrame(fold_results).sort_values("bal_acc")
    print("  Lowest 5 participants by balanced accuracy (potential outliers):")
    for _, r in fold_df.head(5).iterrows():
        print(f"    pid={int(r['pid']):3d}  n={int(r['n']):3d}  bal_acc={r['bal_acc']:.1%}")

    # ── Final fit on all eligible 2022 data ───────────────────────────────────
    print("\nTraining final model on full 2022 eligible dataset …")
    label_encoder = LabelEncoder().fit(y)
    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X, y)
    print("  Done.")

    # In-sample sanity
    y_pred_full = pipeline.predict(X)
    print_results(y, pd.Series(y_pred_full, index=y.index),
                  "FULL TRAINING SET — in-sample sanity (NOT a generalisation metric)")

    # ── Save artifacts ────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline,         ARTIFACTS_DIR / V4_PIPELINE_FILE)
    joblib.dump(label_encoder,    ARTIFACTS_DIR / V4_LABEL_ENCODER_FILE)
    joblib.dump(all_feature_cols, ARTIFACTS_DIR / V4_FEATURE_COLS_FILE)
    joblib.dump({
        "numeric_features":     numeric_features,
        "categorical_features": categorical_features,
        "mucus_features_excluded": sorted(mucus_cols_present),
        "trained_on":           "2022_only",
        "held_out_pids":        sorted(held_out_pids),
        "cv_method":            "LOPO_by_participant",
        "best_model_name":      "lgbm_600_v4_no_mucus",
        "lopo_oof_balanced_acc": float(balanced_accuracy_score(oof_true, oof_pred)),
        "lopo_oof_macro_f1":    float(f1_score(oof_true, oof_pred, average="macro", zero_division=0)),
        "note":                 "Honest metric = lopo_oof_balanced_acc. In-sample number is inflated.",
    }, ARTIFACTS_DIR / V4_METADATA_FILE)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    for f in [V4_PIPELINE_FILE, V4_LABEL_ENCODER_FILE, V4_FEATURE_COLS_FILE, V4_METADATA_FILE]:
        print(f"  {f}")

    print(f"\n{'='*58}")
    print(f"  HONEST GENERALISATION ESTIMATE (LOPO CV)")
    print(f"  Balanced accuracy: {balanced_accuracy_score(oof_true, oof_pred):.1%}")
    print(f"  Macro F1:          {f1_score(oof_true, oof_pred, average='macro', zero_division=0):.3f}")
    print(f"{'='*58}")
    print(f"\nThis is the number to report. NOT the in-sample sanity figure above.")
    print(f"To evaluate on held-out participants: python tests/validate_cycle_v4.py --split held_out")


if __name__ == "__main__":
    main()
