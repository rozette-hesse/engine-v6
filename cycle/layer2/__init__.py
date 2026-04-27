"""
cycle.layer2 — Symptom-Based Phase Classifiers
================================================
Four model variants share the same output dict shape so the fusion layer
can route between them based on the active MODEL_VERSION without any
conditional logic in the caller.

  model_v1  Legacy RandomForest (deprecated)
  model_v3  LightGBM, ordinal severity + cycle context
  model_v4  LightGBM, mucus-clean, LOPO CV (default)
  model_v5  LightGBM, trajectory + 5-day rolling features
"""

from .model_v1 import get_layer2_output      as get_layer2_v1_output  # noqa: F401
from .model_v3 import get_layer2_v3_output                            # noqa: F401
from .model_v4 import get_layer2_v4_output                            # noqa: F401
from .model_v5 import get_layer2_v5_output                            # noqa: F401

__all__ = [
    "get_layer2_v1_output",
    "get_layer2_v3_output",
    "get_layer2_v4_output",
    "get_layer2_v5_output",
]
