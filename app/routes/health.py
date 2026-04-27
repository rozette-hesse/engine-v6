"""GET /health — liveness check."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "version": "6.0"}
