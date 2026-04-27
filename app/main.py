"""
InBalance v6 — FastAPI Application Factory
============================================
Creates the FastAPI app, registers middleware, mounts static assets and
templates, and includes the API routers.

Run:
  python serve.py
  uvicorn app.main:app --reload --port 8000

Environment variables
---------------------
  See .env.example for all supported variables.
  CORS origins are configured via INBALANCE_ALLOWED_ORIGINS.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.config import settings
from app.routes.predict   import router as predict_router
from app.routes.recommend import router as recommend_router
from app.routes.health    import router as health_router

# Resolve paths relative to the project root (one level above app/)
_BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Startup artifact check ────────────────────────────────────────────────────

def _check_artifacts() -> None:
    """Verify that all required .joblib artifacts exist at startup."""
    from cycle.config import ARTIFACTS_DIR, MODEL_VERSION

    _VERSION_FILES = {
        "v1":   ["layer2_vnext_pipeline.joblib", "layer2_vnext_label_encoder.joblib",
                 "layer2_vnext_feature_columns.joblib", "layer2_vnext_metadata.joblib"],
        "v2":   ["layer2_v2_pipeline.joblib", "layer2_v2_label_encoder.joblib",
                 "layer2_v2_feature_columns.joblib", "layer2_v2_metadata.joblib"],
        "v3":   ["layer2_v3_pipeline.joblib", "layer2_v3_label_encoder.joblib",
                 "layer2_v3_feature_columns.joblib", "layer2_v3_metadata.joblib"],
        "v4":   ["layer2_v4_pipeline.joblib", "layer2_v4_label_encoder.joblib",
                 "layer2_v4_feature_columns.joblib", "layer2_v4_metadata.joblib"],
        "vnext": ["layer2_vnext_pipeline.joblib", "layer2_vnext_label_encoder.joblib",
                  "layer2_vnext_feature_columns.joblib", "layer2_vnext_metadata.joblib"],
    }
    _PROXIMITY_FILES = [
        "proximity_pipeline.joblib",
        "proximity_feature_columns.joblib",
    ]

    required = _VERSION_FILES.get(MODEL_VERSION, []) + _PROXIMITY_FILES
    missing = [f for f in required if not (ARTIFACTS_DIR / f).exists()]
    if missing:
        raise RuntimeError(
            f"Missing required artifact files in {ARTIFACTS_DIR}:\n"
            + "\n".join(f"  {f}" for f in missing)
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_artifacts()
    from db.store import init_db, migrate_db
    init_db()
    migrate_db()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="InBalance Engine API",
    version="6.0",
    description="Cycle-phase prediction + personalised wellness recommendations.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(_BASE_DIR / "static")),
    name="static",
)
_templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index(request: Request):
    return _templates.TemplateResponse(name="index.html", request=request)


app.include_router(predict_router)
app.include_router(recommend_router)
app.include_router(health_router)
