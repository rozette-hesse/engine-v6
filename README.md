# InBalance v6

A cycle-aware wellness recommendation engine. Predicts menstrual cycle phase from period history and symptom signals, then scores and ranks personalised exercise, nutrition, sleep, and mental wellness activities.

---

## Architecture

```
inbalance_v6/
├── app/                   FastAPI layer (routes, schemas, app factory)
│   ├── main.py            App creation, middleware, static assets
│   ├── routes/            Thin route handlers — delegate to core/pipeline.py
│   │   ├── predict.py     POST /predict
│   │   ├── recommend.py   POST /recommend
│   │   └── health.py      GET /health
│   └── schemas/
│       ├── enums.py       All input enum classes
│       └── requests.py    UnifiedRequest, ManualRequest
│
├── core/                  Business logic (no FastAPI)
│   ├── pipeline.py        Orchestrates predict + recommend flows
│   └── transformers.py    Converts severity strings → engine inputs
│
├── cycle/                 Cycle prediction engine
│   ├── fusion.py          Multi-layer fusion orchestrator
│   ├── layer1_period.py   Period history → timing prior
│   ├── layer2/            Symptom-based ML classifiers
│   │   ├── base.py        Shared helpers and artifact cache
│   │   ├── features_v3.py Ordinal feature builder (v3/v4)
│   │   ├── model_v1.py    Legacy RandomForest (deprecated)
│   │   ├── model_v3.py    LightGBM with ordinal severity + cycle context
│   │   └── model_v4.py    LightGBM mucus-clean, LOPO CV (default)
│   ├── layer3_timing.py   Timing narrative (human-readable notes)
│   ├── phase_resolver.py  4-phase engine → 5-phase recommender bridge
│   ├── proximity.py       Period proximity classifier
│   └── config.py          Artifact paths, phase constants, model version
│
├── recommender/           Recommendation engine
│   ├── engine.py          Graph-based spreading activation scorer
│   └── scales.py          Ordinal rank tables for all input signals
│
├── data/                  Static domain knowledge (edited by experts)
│   ├── activities.py      Activity inventory (~100 items across 4 domains)
│   └── rules.py           DELTA + BLOCK rules driving the recommender
│
├── tests/
│   ├── validate.py        25 golden clinical scenarios (primary CI check)
│   ├── validate_graph.py  Runs all scenarios against the graph recommender
│   └── validate_report.py HTML report generator
│
├── scripts/               ML training scripts (not in CI)
├── artifacts/             Binary model artifacts (.joblib)
├── static/                Frontend JS + CSS
├── templates/             Jinja2 HTML template
└── serve.py               Development server launcher
```

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| `cycle/layer2/` sub-package | Three model variants share a `base.py` with a unified artifact cache. Each variant is an isolated module. |
| `core/pipeline.py` | Route handlers stay thin; all orchestration lives in one testable place. |
| `recommender/scales.py` | Ordinal rank tables extracted from the engine so they can be imported by tests without loading the full engine. |
| `cycle/phase_resolver.py` | Moved from `engine/` — it belongs with the cycle package since it bridges cycle output to the recommender. |
| Dead code removed | `engine/graph_recommender.py` (compatibility shim) and `engine/recommender_flat_deprecated.py` are not present in v6. The latter is in `archive/` for reference only. |

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure artifacts are present
ls artifacts/              # should list *.joblib files
# If artifacts are missing, copy from v5:
# cp ../inbalance_v5/artifacts/*.joblib artifacts/
```

---

## Running the app

```bash
# Development (hot reload)
python serve.py

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Open http://localhost:8000 for the web UI.

---

## API

### `POST /predict`
Full pipeline: cycle prediction → phase resolution → recommendations.

```json
{
  "period_starts": ["2026-03-15", "2026-02-14"],
  "ENERGY": "Low",
  "STRESS": "Low",
  "MOOD": "Low",
  "CRAMP": "Mild",
  "HEADACHE": 0,
  "top_n": 3
}
```

### `POST /recommend`
Manual mode — supply phase directly, skip cycle prediction.

```json
{
  "PHASE": "MEN",
  "ENERGY": "Low",
  "STRESS": "Low",
  "MOOD": "Low",
  "CRAMP": "Mild",
  "HEADACHE": 0
}
```

### `GET /health`
```json
{"status": "ok", "version": "6.0"}
```

---

## Validation

```bash
# Run all 25 clinical scenarios against the recommender (primary CI check)
python tests/validate_graph.py

# Run the recommender scenarios directly
python tests/validate.py
```

All 25 scenarios must pass before merging changes to `data/rules.py`,
`data/activities.py`, or `recommender/`.

---

## Model versions

The active cycle prediction model is controlled by the `INBALANCE_MODEL_VERSION`
environment variable (default: `v4`).

| Version | Description |
|---------|-------------|
| `v4`    | LightGBM, mucus-clean features, LOPO CV — **default** |
| `v3`    | LightGBM, ordinal severity + cycle context |
| `v1`    | RandomForest, binary symptoms — deprecated, no training script |

```bash
INBALANCE_MODEL_VERSION=v3 python serve.py
```

Artifact directory can be overridden:
```bash
INBALANCE_ARTIFACTS_DIR=/path/to/artifacts python serve.py
```

---

## Adding or changing recommendations

1. Edit `data/rules.py` (DELTA for scoring, BLOCK for hard guards).
2. Edit `data/activities.py` if adding new activities or tags.
3. Run `python tests/validate_graph.py` — all 25 scenarios must pass.
4. If a scenario needs updating due to an intentional behaviour change, update
   `tests/validate.py` with a comment explaining the clinical rationale.

---

## Project conventions

- Route handlers contain only HTTP concerns (parse request, return response).
- All business logic lives in `core/` or the engine packages.
- `data/rules.py` and `data/activities.py` are maintained by domain experts; do not reformat them.
- Artifact files in `artifacts/` are binary and generated; do not commit changes to them.
- `scripts/` is for one-off tooling — not imported by the app.
