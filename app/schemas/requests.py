"""
Request models for the InBalance API.

UnifiedRequest  — full pipeline: cycle prediction → recommendations
ManualRequest   — skip cycle prediction, supply phase directly
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from .enums import (
    Phase, EnergyLevel, StressLevel, MoodLevel, CrampLevel,
    MoodSwingLevel, CravingsLevel, FlowLevel, TendernessLevel,
    IndigestionLevel, BrainFogLevel, BloatingLevel, SleepQuality,
    IrritabilityLevel, CervicalMucus,
)


class UnifiedRequest(BaseModel):
    """
    Full pipeline request.

    Cycle inputs are used by the cycle engine (period history, mucus, flow,
    appetite, exercise).  Symptom signals are consumed by both the cycle
    engine (as binary presence/absence) and the recommendation engine (as
    severity levels).
    """
    # User identity — required for log persistence and history retrieval.
    # Pass any stable string ID from your auth provider (Firebase UID, UUID, etc.).
    # Omit or set null to disable history (stateless single-shot mode).
    user_id:  Optional[str] = None
    log_date: Optional[str] = Field(default=None, description="Date of this log (YYYY-MM-DD). Defaults to today.")

    # Cycle timing inputs
    period_starts:       List[str]         = Field(..., description="Period start dates (YYYY-MM-DD)")
    period_start_logged: bool              = False
    cervical_mucus:      CervicalMucus     = CervicalMucus.unknown
    appetite:            int               = Field(default=0, ge=0, le=10)
    exercise_level:      int               = Field(default=0, ge=0, le=10)

    # Symptom signals (severity-rated)
    ENERGY:            EnergyLevel        = EnergyLevel.medium
    STRESS:            StressLevel        = StressLevel.low
    MOOD:              MoodLevel          = MoodLevel.medium
    CRAMP:             CrampLevel         = CrampLevel.none
    HEADACHE:          int                = Field(default=0, ge=0, le=10)
    BLOATING:          BloatingLevel      = BloatingLevel.none
    SLEEP_QUALITY:     SleepQuality       = SleepQuality.good
    IRRITABILITY:      IrritabilityLevel  = IrritabilityLevel.none
    MOOD_SWING:        MoodSwingLevel     = MoodSwingLevel.none
    CRAVINGS:          CravingsLevel      = CravingsLevel.none
    FLOW:              FlowLevel          = FlowLevel.none
    BREAST_TENDERNESS: TendernessLevel    = TendernessLevel.none
    INDIGESTION:       IndigestionLevel   = IndigestionLevel.none
    BRAIN_FOG:         BrainFogLevel      = BrainFogLevel.none

    top_n: int = 3


class ManualRequest(BaseModel):
    """
    Manual recommendation request.

    The caller supplies PHASE directly; cycle prediction is skipped.
    Useful when users know their phase or for testing specific phase scenarios.
    """
    PHASE:             Phase
    ENERGY:            EnergyLevel
    STRESS:            StressLevel
    MOOD:              MoodLevel
    CRAMP:             CrampLevel
    HEADACHE:          int               = Field(default=0, ge=0, le=10)
    BLOATING:          BloatingLevel     = BloatingLevel.none
    SLEEP_QUALITY:     SleepQuality      = SleepQuality.good
    IRRITABILITY:      IrritabilityLevel = IrritabilityLevel.none
    MOOD_SWING:        MoodSwingLevel    = MoodSwingLevel.none
    CRAVINGS:          CravingsLevel     = CravingsLevel.none
    FLOW:              FlowLevel         = FlowLevel.none
    BREAST_TENDERNESS: TendernessLevel   = TendernessLevel.none
    INDIGESTION:       IndigestionLevel  = IndigestionLevel.none
    BRAIN_FOG:         BrainFogLevel     = BrainFogLevel.none

    top_n: int = 3
