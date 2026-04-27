"""
Cycle engine configuration.

Artifact paths, phase constants, model version, and fusion weights are
all defined here so the rest of the cycle package has a single source
of truth for tunable values.

Model version
-------------
  v1 (vnext)  legacy RandomForest — binary symptoms only, deprecated
  v2          RF + 2 cycle-day features
  v3          LightGBM, ordinal severity + cycle context  (prev default)
  v4          LightGBM, mucus-clean, LOPO CV              (current default)
  v5          LightGBM, trajectory + rolling features, 5-day history (experimental — underperforms v4 on held-out)

Override at runtime via the INBALANCE_MODEL_VERSION environment variable.
Override artifact directory via INBALANCE_ARTIFACTS_DIR.

Runtime values are sourced from core.config.settings (pydantic-settings),
which validates them at startup and reads from .env if present.
Standalone scripts that don't boot the app fall back to raw env-var reads.
"""

from pathlib import Path

try:
    from core.config import settings as _settings
    ARTIFACTS_DIR: Path = _settings.artifacts_dir
    MODEL_VERSION: str  = _settings.model_version.lower()
except Exception:
    # Fallback for standalone scripts / import-time circular deps
    import os
    _BASE_DIR     = Path(__file__).resolve().parent.parent
    ARTIFACTS_DIR = Path(os.environ.get("INBALANCE_ARTIFACTS_DIR", str(_BASE_DIR / "artifacts")))
    MODEL_VERSION = os.environ.get("INBALANCE_MODEL_VERSION", "v4").lower()

# ── Phase constants ───────────────────────────────────────────────────────────

# All 4 phases output by the cycle engine
PHASES = ["Menstrual", "Follicular", "Fertility", "Luteal"]

# Non-menstrual phases predicted by Layer 2 and the fusion layer
NON_MENSTRUAL_PHASES = ["Follicular", "Fertility", "Luteal"]

# ── Symptom vocabulary ────────────────────────────────────────────────────────

SUPPORTED_SYMPTOMS = [
    "headaches",
    "cramps",
    "sorebreasts",
    "fatigue",
    "sleepissue",
    "moodswing",
    "stress",
    "foodcravings",
    "indigestion",
    "bloating",
]

MUCUS_OPTIONS = ["dry", "sticky", "creamy", "eggwhite", "watery", "unknown"]

# ── Fusion weights ────────────────────────────────────────────────────────────
# v1/v2: L1 carries less weight because it has no cycle-day awareness in L2.
LAYER1_WEIGHT    = 0.2
LAYER2_WEIGHT    = 0.8

# v3/v4: L2 already bakes in cycle position, so L1 prior carries even less.
LAYER1_WEIGHT_V3 = 0.10
LAYER2_WEIGHT_V3 = 0.90

# ── Artifact filenames ────────────────────────────────────────────────────────

# v1 (vnext) — deprecated; no recoverable training script
LAYER2_PIPELINE_FILE        = "layer2_vnext_pipeline.joblib"
LAYER2_LABEL_ENCODER_FILE   = "layer2_vnext_label_encoder.joblib"
LAYER2_FEATURE_COLUMNS_FILE = "layer2_vnext_feature_columns.joblib"
LAYER2_METADATA_FILE        = "layer2_vnext_metadata.joblib"

# v2
LAYER2_V2_PIPELINE_FILE        = "layer2_v2_pipeline.joblib"
LAYER2_V2_LABEL_ENCODER_FILE   = "layer2_v2_label_encoder.joblib"
LAYER2_V2_FEATURE_COLUMNS_FILE = "layer2_v2_feature_columns.joblib"
LAYER2_V2_METADATA_FILE        = "layer2_v2_metadata.joblib"

# v3
LAYER2_V3_PIPELINE_FILE        = "layer2_v3_pipeline.joblib"
LAYER2_V3_LABEL_ENCODER_FILE   = "layer2_v3_label_encoder.joblib"
LAYER2_V3_FEATURE_COLUMNS_FILE = "layer2_v3_feature_columns.joblib"
LAYER2_V3_METADATA_FILE        = "layer2_v3_metadata.joblib"

# v4 — mucus-clean, LOPO CV
LAYER2_V4_PIPELINE_FILE        = "layer2_v4_pipeline.joblib"
LAYER2_V4_LABEL_ENCODER_FILE   = "layer2_v4_label_encoder.joblib"
LAYER2_V4_FEATURE_COLUMNS_FILE = "layer2_v4_feature_columns.joblib"
LAYER2_V4_METADATA_FILE        = "layer2_v4_metadata.joblib"

# v5 — trajectory-aware, 5-day history window
LAYER2_V5_PIPELINE_FILE        = "layer2_v5_pipeline.joblib"
LAYER2_V5_LABEL_ENCODER_FILE   = "layer2_v5_label_encoder.joblib"
LAYER2_V5_FEATURE_COLUMNS_FILE = "layer2_v5_feature_columns.joblib"
LAYER2_V5_METADATA_FILE        = "layer2_v5_metadata.joblib"
