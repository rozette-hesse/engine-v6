"""POST /recommend — manual phase override, skip cycle prediction."""

from fastapi import APIRouter
from app.schemas.requests import ManualRequest
from core.pipeline import run_recommend

router = APIRouter()


@router.post("/recommend")
def get_recommendations(payload: ManualRequest):
    """
    Manual mode: caller supplies the phase directly.
    Useful for testing specific phase scenarios or when cycle history
    is unavailable and the user knows their current phase.
    """
    return run_recommend(payload)
