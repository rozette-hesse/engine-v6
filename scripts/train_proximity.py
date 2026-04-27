#!/usr/bin/env python3
"""
Train period proximity classifier.

Predicts how many days until the next period from symptoms alone —
no cycle-day, no timing features. Designed for irregular cycle users
where timing estimates are unreliable.

Labels (days until next period):
  soon  — 0–3 days
  near  — 4–7 days
  far   — 8+ days

Protocol mirrors v4:
  - 2022 data only
  - Held-out participants excluded
  - LOPO CV for honest estimate
  - No mucus features

Run:  python train_proximity.py
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
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(ROOT))

from cycle.config import ARTIFACTS_DIR
from cycle.layer2.features_v3 import SUPPORTED_SYMPTOMS, to_ordinal

HELD_OUT_FILE = ROOT / "scripts" / "held_out_participants.json"
PROXIMITY_PIPELINE_FILE      = "proximity_pipeline.joblib"
PROXIMITY_FEATURE_COLS_FILE  = "proximity_feature_columns.joblib"
PROXIMITY_METADATA_FILE      = "proximity_metadata.joblib"

LABELS = ["soon", "not_soon"]

PROXIMITY_DAYS = 7   # "soon" = period arriving within this many days


def days_to_label(days: int) -> str:
    return "soon" if days <= PROXIMITY_DAYS else "not_soon"


def build_symptom_features(row) -> dict:
    """Pure symptom feature row — no cycle day, no timing."""
    feat = {}
    for sym in SUPPORTED_SYMPTOMS:
        feat[f"{sym}_ord"] = to_ordinal(row.get(sym))
    feat["appetite_ord"]      = to_ordinal(row.get("appetite"))
    feat["exerciselevel_ord"] = to_ordinal(row.get("exerciselevel"))
    # Total symptom load
    feat["total_severity"] = sum(
        int(feat[f"{sym}_ord"] or 0) for sym in SUPPORTED_SYMPTOMS
    )
    return feat


def build_dataset(df: pd.DataFrame, held_out_pids: set):
    BASE_DATE = datetime(2022, 1, 1)
    records = []
    skipped = 0

    df_2022 = df[
        (df["study_interval"] == 2022) & (~df["id"].isin(held_out_pids))
    ]
    print(f"  Training pool: {len(df_2022)} rows | "
          f"{df_2022['id'].nunique()} participants")

    for pid, grp in df_2022.groupby("id", sort=True):
        grp = grp.sort_values("day_in_study").reset_index(drop=True)
        grp["date"] = grp["day_in_study"].apply(
            lambda d: BASE_DATE + timedelta(days=int(d) - 1)
        )

        # Find period start dates (first Menstrual day of each run)
        period_starts = []
        prev_men = False
        for _, row in grp.iterrows():
            is_men = str(row["phase"]) == "Menstrual"
            if is_men and not prev_men:
                period_starts.append(row["date"])
            prev_men = is_men

        if len(period_starts) < 2:
            skipped += len(grp)
            continue

        for _, row in grp.iterrows():
            phase_truth = str(row["phase"])
            if phase_truth in ("Menstrual", "nan", "None", ""):
                continue  # skip menstrual days — no "days until period" signal

            today = row["date"]
            # Find the next period start after today
            future = [s for s in period_starts if s > today]
            if not future:
                continue
            days_until = (min(future) - today).days
            if days_until < 0:
                continue

            feat = build_symptom_features(row)
            feat["_label"]       = days_to_label(days_until)
            feat["_participant"] = pid
            records.append(feat)

    print(f"  Skipped {skipped} rows (insufficient history or Menstrual)")
    print(f"  Feature rows collected: {len(records)}")
    data_df   = pd.DataFrame(records)
    y         = data_df.pop("_label")
    groups    = data_df.pop("_participant")
    return data_df, y, groups


def build_pipeline(feature_cols):
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), feature_cols),
    ])
    clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=10,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_lambda=1.5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return SKPipeline([("preprocessor", preprocessor), ("model", clf)])


def print_results(y_true, y_pred, title):
    sep = "=" * 58
    total   = len(y_true)
    correct = (y_true == y_pred).sum()
    print(f"\n{sep}\n  {title}\n{sep}")
    print(f"  Accuracy:          {correct}/{total} = {correct/total:.1%}")
    print(f"  Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.1%}")
    print(f"  Macro F1:          {f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}")
    print("\n  Per-label recall:")
    for lbl in LABELS:
        mask = y_true == lbl
        if mask.sum() == 0:
            continue
        rec = (y_pred[mask] == lbl).mean()
        print(f"    {lbl:6s}  n={mask.sum():4d}  recall={rec:.1%}")
    print("\n  Confusion matrix (rows=truth, cols=pred):")
    print(pd.crosstab(y_true, y_pred, rownames=["truth"], colnames=["pred"]).to_string())


def main():
    warnings.filterwarnings("ignore")

    if HELD_OUT_FILE.exists():
        with open(HELD_OUT_FILE) as f:
            held_out_pids = set(json.load(f)["held_out_participant_ids"])
        print(f"Held-out excluded: {sorted(held_out_pids)}")
    else:
        held_out_pids = set()

    csv_path = ROOT / "data" / "mcphases_with_estimated_cervical_mucus_v3.csv"
    print(f"\nLoading {csv_path.name} …")
    df = pd.read_csv(csv_path)

    print("\nBuilding proximity feature matrix (symptom-only) …")
    X, y, groups = build_dataset(df, held_out_pids)

    feature_cols = list(X.columns)
    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Label distribution:\n{y.value_counts().to_string()}")

    # LOPO CV
    print(f"\nRunning LOPO CV ({groups.nunique()} folds) …")
    logo = LeaveOneGroupOut()
    oof_true, oof_pred = [], []

    for fold, (tr, va) in enumerate(logo.split(X, y, groups=groups), 1):
        pipe = build_pipeline(feature_cols)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        preds = pipe.predict(X.iloc[va])
        oof_true.extend(y.iloc[va].tolist())
        oof_pred.extend(preds.tolist())
        if fold % 5 == 0 or fold == groups.nunique():
            acc = (y.iloc[va].values == preds).mean()
            print(f"  Fold {fold:2d}/{groups.nunique()}  acc={acc:.1%}")

    oof_true = pd.Series(oof_true)
    oof_pred = pd.Series(oof_pred)
    print_results(oof_true, oof_pred, "LOPO CV — HONEST ESTIMATE")

    # Final fit
    print("\nTraining final model on full eligible dataset …")
    pipeline = build_pipeline(feature_cols)
    pipeline.fit(X, y)

    # Save
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline,     ARTIFACTS_DIR / PROXIMITY_PIPELINE_FILE)
    joblib.dump(feature_cols, ARTIFACTS_DIR / PROXIMITY_FEATURE_COLS_FILE)
    joblib.dump({
        "feature_cols":        feature_cols,
        "labels":              LABELS,
        "label_map":           {"soon": f"0-{PROXIMITY_DAYS} days", "not_soon": f"{PROXIMITY_DAYS+1}+ days"},
        "proximity_days":      PROXIMITY_DAYS,
        "trained_on":          "2022_only_no_timing",
        "held_out_pids":       sorted(held_out_pids),
        "cv_method":           "LOPO_by_participant",
        "lopo_balanced_acc":   float(balanced_accuracy_score(oof_true, oof_pred)),
        "lopo_macro_f1":       float(f1_score(oof_true, oof_pred, average="macro", zero_division=0)),
        "lopo_soon_recall":    float((oof_pred[oof_true == "soon"] == "soon").mean()),
    }, ARTIFACTS_DIR / PROXIMITY_METADATA_FILE)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    bal = balanced_accuracy_score(oof_true, oof_pred)
    f1  = f1_score(oof_true, oof_pred, average="macro", zero_division=0)
    print(f"\n{'='*58}")
    print(f"  HONEST ESTIMATE (LOPO CV)")
    print(f"  Balanced accuracy: {bal:.1%}")
    print(f"  Macro F1:          {f1:.3f}")
    print(f"{'='*58}")


if __name__ == "__main__":
    main()
