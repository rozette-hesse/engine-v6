# Scripts

One-off ML training scripts, evaluation utilities, and experiment tools.

These are **not** part of the production app and are **not** run in CI.
Run everything from the project root.

## Training

| Script | Purpose |
|--------|---------|
| `train_layer2_v3.py` | Train the v3 LightGBM Layer 2 model (ordinal severity + cycle context) |
| `train_layer2_v4.py` | Train the v4 mucus-clean LightGBM model (no label leakage, LOPO CV) |
| `train_proximity.py` | Train the period-proximity classifier (symptom-only, no timing) |

## Evaluation

| Script | Purpose |
|--------|---------|
| `evaluate_cycle.py` | Dataset-level accuracy — replays all participants with rolling history |
| `validate_cycle.py` | Golden scenario regression tests (delegates to `tests/validate_cycle.py`) |
| `compare_models.py`  | Side-by-side KPI comparison of v1/v2/v3/v4 through the fusion pipeline |

## Smoke tests

| Script | Purpose |
|--------|---------|
| `test_cold_start.py` | Smoke-test the cold-start (no history) code path |

## Setup

Data files live in `data/`. Trained artifacts are written to `artifacts/`.

```bash
# From the project root:
python scripts/train_layer2_v4.py
python scripts/evaluate_cycle.py
python scripts/evaluate_cycle.py --held-out   # held-out participants only
```

## Data

`data/mcphases_with_estimated_cervical_mucus_v3.csv` — 2022 clinical study wave.
Not committed to version control (see `.gitignore`).

Training uses 2022 wave only. 2024 wave has 99.4% symptom missingness.
Participants in `held_out_participants.json` are excluded from training
and reserved for a single final evaluation.
