#!/usr/bin/env python3
"""
Train Layer 2 v3 cycle phase model.

Improvements over v2
────────────────────
1.  Preserves ordinal symptom severity (0..5) instead of binarising at >=3.
    The v2 model collapses every "Low/Moderate/High/Very High" symptom into a
    single bit, throwing away most of the signal in the dataset.

2.  Severity-weighted group features (pain_ord_sum, energy_ord_max, …) plus
    a total_severity load score.

3.  Extra cycle-timing features:
        cycle_day_norm, days_from_ovulation, is_past_ovulation,
        cycle_length_cv, n_periods_logged                     (kept from v2)
        cycle_day_sin, cycle_day_cos                          (cyclical)
        days_until_next_period_norm                           (forward)

4.  Layer-1 priors injected as features (l1_prior_follicular / fertility /
    luteal + forecast-confidence flags) so the model can lean on cycle-day
    evidence directly instead of relying on the fusion layer.

5.  Explicit bleeding history: bleeding_lag1, bleeding_lag2, bleeding_streak3
    so the model learns when a stray flow log indicates ongoing menses.

6.  LightGBM classifier with mild regularisation, trained on the FULL dataset
    (2022 + 2024). Honest out-of-sample metric comes from a participant-grouped
    5-fold cross-validation that runs before the final fit.

Artifacts:
    artifacts/layer2_v3_pipeline.joblib
    artifacts/layer2_v3_label_encoder.joblib
    artifacts/layer2_v3_feature_columns.joblib
    artifacts/layer2_v3_metadata.joblib

Run:  python train_layer2_v3.py
"""

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
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupKFold
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

V3_PIPELINE_FILE      = "layer2_v3_pipeline.joblib"
V3_LABEL_ENCODER_FILE = "layer2_v3_label_encoder.joblib"
V3_FEATURE_COLS_FILE  = "layer2_v3_feature_columns.joblib"
V3_METADATA_FILE      = "layer2_v3_metadata.joblib"

MUCUS_MAP = {"egg_white": "eggwhite"}


# ── CSV → severity dict ─────────────────────────────────────────────────────
def row_to_severities(row) -> dict:
    return {sym: to_ordinal(row.get(sym)) for sym in SUPPORTED_SYMPTOMS}


def row_to_mucus(row) -> str:
    val = row.get("cervical_mucus_estimated_type_v3")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "unknown"
    s = str(val).strip().lower()
    s = MUCUS_MAP.get(s, s)
    return s if s in {"dry", "sticky", "creamy", "watery", "eggwhite"} else "unknown"


def row_to_flow(row) -> str:
    val = row.get("flow_volume", "none")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "none"
    return str(val).strip()


# ── Dataset builder ─────────────────────────────────────────────────────────
def build_dataset(df: pd.DataFrame):
    BASE_DATES = {2022: datetime(2022, 1, 1), 2024: datetime(2024, 1, 1)}

    records = []
    skipped = 0

    for (pid, year), grp in df.groupby(["id", "study_interval"], sort=True):
        grp = grp.sort_values("day_in_study").reset_index(drop=True)
        base = BASE_DATES.get(int(year), datetime(2022, 1, 1))
        grp["date"] = grp["day_in_study"].apply(
            lambda d: (base + timedelta(days=int(d) - 1)).strftime("%Y-%m-%d")
        )

        known_starts   = []
        prev_men       = False
        history_buffer = []

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

            severities = row_to_severities(row)
            mucus      = row_to_mucus(row)
            appetite   = to_ordinal(row.get("appetite"))
            exerciselevel = to_ordinal(row.get("exerciselevel"))
            flow       = row_to_flow(row)

            recent_daily_logs = [
                {
                    "symptoms":       severities_to_binary_symptoms(h["severities"]),
                    "cervical_mucus": h["mucus"],
                    "appetite":       h["appetite"],
                    "exerciselevel":  h["exerciselevel"],
                    "flow":           h["flow"],
                }
                for h in history_buffer[-2:]
            ]

            layer1 = get_layer1_output(list(known_starts), today=today)

            # Skip null / Menstrual rows — the L2 model only learns the
            # 3 non-menstrual classes (the fusion layer handles Menstrual).
            if phase_truth in ("nan", "None", "") or pd.isna(phase_truth):
                history_buffer.append({"severities": severities, "mucus": mucus,
                                       "appetite": appetite, "exerciselevel": exerciselevel,
                                       "flow": flow})
                prev_men = is_men
                continue

            if is_period_day1 or (layer1.get("cycle_day") and layer1["cycle_day"] <= 5) or is_men:
                history_buffer.append({"severities": severities, "mucus": mucus,
                                       "appetite": appetite, "exerciselevel": exerciselevel,
                                       "flow": flow})
                prev_men = is_men
                continue

            feat = build_v3_feature_row(
                symptom_severities=severities,
                cervical_mucus=mucus,
                appetite=appetite,
                exerciselevel=exerciselevel,
                flow=flow,
                recent_daily_logs=recent_daily_logs,
                layer1=layer1,
                known_starts=list(known_starts),
            )
            feat["_label"]       = phase_truth
            feat["_year"]        = int(year)
            feat["_participant"] = pid
            records.append(feat)

            history_buffer.append({"severities": severities, "mucus": mucus,
                                   "appetite": appetite, "exerciselevel": exerciselevel,
                                   "flow": flow})
            prev_men = is_men

    print(f"  Skipped {skipped} rows (no prior period history)")
    print(f"  Feature rows collected: {len(records)}")

    data_df = pd.DataFrame(records)
    y           = data_df.pop("_label")
    groups_year = data_df.pop("_year")
    groups_pid  = data_df.pop("_participant")
    return data_df, y, groups_year, groups_pid


# ── Pipeline ────────────────────────────────────────────────────────────────
def build_pipeline_lgbm(numeric_features, categorical_features):
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
    return SKPipeline([
        ("preprocessor", preprocessor),
        ("model",        clf),
    ])


def print_results(y_true, y_pred, title: str):
    phases = ["Follicular", "Fertility", "Luteal"]
    sep = "=" * 55
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


def main():
    warnings.filterwarnings("ignore")
    csv_path = ROOT / "data" / "mcphases_with_estimated_cervical_mucus_v3.csv"
    print(f"Loading {csv_path.name} …")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows | {df['id'].nunique()} participants\n")

    print("Building feature matrix (v3 — ordinal severities + cycle features) …")
    X, y, groups_year, groups_pid = build_dataset(df)

    categorical_features = ["exerciselevel", "appetite", "mucus_type"]
    numeric_features     = [c for c in X.columns if c not in categorical_features]
    all_feature_cols     = numeric_features + categorical_features

    print(f"\n  Total features: {len(all_feature_cols)}")
    print(f"  Numeric: {len(numeric_features)}   Categorical: {len(categorical_features)}")

    new_v3_features = [
        c for c in all_feature_cols if (
            c.endswith("_ord") or c.endswith("_ord_sum") or c.endswith("_ord_max") or
            c.endswith("_ord_mean") or c == "total_severity" or
            c.startswith("l1_prior") or c.startswith("l1_forecast") or
            c.startswith("cycle_day_") or c == "days_until_next_period_norm" or
            c.startswith("bleeding_lag") or c == "bleeding_streak3"
        )
    ]
    print(f"  New v3 feature columns: {len(new_v3_features)}")

    X = X[all_feature_cols]
    print(f"\n  Full dataset: {len(X):,} rows  {dict(y.value_counts())}")

    # ── Participant-grouped 5-fold CV for honest out-of-sample estimate ─────
    print("\nRunning 5-fold participant-grouped CV …")
    gkf = GroupKFold(n_splits=5)
    oof_true, oof_pred = [], []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups_pid), 1):
        pipe = build_pipeline_lgbm(numeric_features, categorical_features)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        preds = pipe.predict(X.iloc[va])
        oof_true.extend(y.iloc[va].tolist())
        oof_pred.extend(preds.tolist())
        fold_acc = (y.iloc[va].values == preds).mean()
        print(f"  Fold {fold}: {fold_acc:.1%}")

    oof_true = pd.Series(oof_true)
    oof_pred = pd.Series(oof_pred)
    print_results(oof_true, oof_pred, "5-FOLD CV — participant-grouped (out-of-sample)")

    # ── Final fit on the full dataset ───────────────────────────────────────
    print("Training final LightGBM on full dataset (2022 + 2024) …")
    label_encoder = LabelEncoder().fit(y)

    pipeline = build_pipeline_lgbm(numeric_features, categorical_features)
    pipeline.fit(X, y)
    print("  Done.\n")

    y_pred_full = pipeline.predict(X)
    print_results(y, pd.Series(y_pred_full, index=y.index), "FULL DATASET (in-sample sanity)")

    # ── Save ────────────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline,         ARTIFACTS_DIR / V3_PIPELINE_FILE)
    joblib.dump(label_encoder,    ARTIFACTS_DIR / V3_LABEL_ENCODER_FILE)
    joblib.dump(all_feature_cols, ARTIFACTS_DIR / V3_FEATURE_COLS_FILE)
    joblib.dump({
        "numeric_features":     numeric_features,
        "categorical_features": categorical_features,
        "new_v3_features":      new_v3_features,
        "best_model_name":      "lgbm_600_v3_ordinal",
        "trained_on":           "2022+2024",
        "preserves_ordinal":    True,
    }, ARTIFACTS_DIR / V3_METADATA_FILE)

    print(f"\nArtifacts saved: {V3_PIPELINE_FILE}, etc.")


if __name__ == "__main__":
    main()
