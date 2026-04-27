"""POST /predict — full cycle + recommendation pipeline."""

from fastapi import APIRouter
from app.schemas.requests import UnifiedRequest
from core.pipeline import run_prediction

router = APIRouter()


@router.post("/predict")
def predict(payload: UnifiedRequest):
    """
    Full pipeline:
      1. Cycle engine predicts phase from period history + symptoms
      2. Phase resolver maps 4-phase → 5-phase code
      3. Recommendation engine scores and ranks activities
      4. Period-proximity classifier adds a pre-menstrual signal
    """
    return run_prediction(payload)
