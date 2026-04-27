"""
Layer 1 — Period History Model
================================
Uses logged period start dates to compute cycle timing features:
  - Weighted average cycle length (recent periods weighted more heavily)
  - Current cycle day
  - Phase probability prior from cycle day position
  - Predicted next period, fertile window, and ovulation date
  - Regularity and forecast confidence ratings

This layer produces a timing prior that the fusion layer combines with
Layer 2's symptom-based signal.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .config import PHASES
from .utils import parse_date, days_between, normalize_probs


def compute_cycle_lengths(period_starts: List[str]) -> List[int]:
    """Compute day-count gaps between consecutive period starts."""
    dates = sorted(parse_date(d) for d in period_starts)
    if len(dates) < 2:
        return []
    return [days_between(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]


def weighted_recent_cycle_length(lengths: List[int]) -> Optional[float]:
    """Return a recency-weighted average cycle length. More recent cycles carry higher weight."""
    if not lengths:
        return None
    weights = list(range(1, len(lengths) + 1))
    return sum(w * x for w, x in zip(weights, lengths)) / sum(weights)


def estimate_cycle_day(period_starts: List[str], today: Optional[str] = None) -> Optional[int]:
    """Day 1 = first day of the most recent period."""
    if not period_starts:
        return None
    latest    = max(parse_date(d) for d in period_starts)
    today_dt  = parse_date(today) if today else datetime.today()
    return max((today_dt - latest).days + 1, 1)


def phase_probs_from_cycle_day(
    cycle_day: Optional[int],
    cycle_length: Optional[float],
) -> Dict[str, float]:
    """
    Build a timing-based phase probability prior.

    Uses calendar heuristics (5-day menstrual window, ovulation = cycle_length - 14)
    to assign a dominant phase with residual probability for other phases.
    """
    if cycle_day is None or cycle_length is None:
        return {p: 0.25 for p in PHASES}

    ovulation_day = round(cycle_length - 14)

    probs = {"Menstrual": 0.05, "Follicular": 0.10, "Fertility": 0.10, "Luteal": 0.10}

    if 1 <= cycle_day <= 5:
        probs["Menstrual"] += 0.75
    elif 6 <= cycle_day <= max(ovulation_day - 4, 6):
        probs["Follicular"] += 0.70
    elif max(ovulation_day - 3, 1) <= cycle_day <= ovulation_day + 2:
        probs["Fertility"] += 0.75
    elif cycle_day > ovulation_day + 2:
        probs["Luteal"] += 0.75

    return normalize_probs(probs)


def get_regularity_status(cycle_lengths: List[int]) -> str:
    """Classify cycle regularity based on standard deviation of lengths."""
    if len(cycle_lengths) < 3:
        return "limited_history"
    mean_len = sum(cycle_lengths) / len(cycle_lengths)
    std      = (sum((x - mean_len) ** 2 for x in cycle_lengths) / len(cycle_lengths)) ** 0.5
    if std <= 2:
        return "regular"
    if std <= 5:
        return "some_variation"
    return "irregular"


def get_forecast_confidence(cycle_lengths: List[int]) -> str:
    """Rate forecast reliability: high / medium / low based on cycle variability."""
    if len(cycle_lengths) < 3:
        return "low"
    mean_len = sum(cycle_lengths) / len(cycle_lengths)
    std      = (sum((x - mean_len) ** 2 for x in cycle_lengths) / len(cycle_lengths)) ** 0.5
    if std <= 2:
        return "high"
    if std <= 5:
        return "medium"
    return "low"


def get_layer1_output(period_starts: List[str], today: Optional[str] = None) -> Dict[str, object]:
    """
    Compute all Layer 1 cycle timing outputs from period history.

    Returns a dict with:
      cycle_lengths, estimated_cycle_length, cycle_day,
      predicted_next_period, next_period_window,
      possible_ovulation_day, possible_ovulation_date,
      fertile_window, regularity_status, forecast_confidence,
      phase_probs
    """
    cycle_lengths    = compute_cycle_lengths(period_starts)
    avg_cycle_length = weighted_recent_cycle_length(cycle_lengths)
    cycle_day        = estimate_cycle_day(period_starts, today=today)

    predicted_next_period = None
    next_period_window    = None
    ovulation_day         = None
    possible_ovulation_date = None
    fertile_window        = None

    if period_starts and avg_cycle_length is not None:
        latest = max(parse_date(d) for d in period_starts)
        predicted_next_period_dt = latest + timedelta(days=round(avg_cycle_length))
        predicted_next_period    = predicted_next_period_dt.strftime("%Y-%m-%d")

        confidence  = get_forecast_confidence(cycle_lengths)
        window_size = 2 if confidence == "high" else 4 if confidence == "medium" else 6
        next_period_window = {
            "start": (predicted_next_period_dt - timedelta(days=window_size)).strftime("%Y-%m-%d"),
            "end":   (predicted_next_period_dt + timedelta(days=window_size)).strftime("%Y-%m-%d"),
        }

        ovulation_day           = round(avg_cycle_length - 14)
        possible_ovulation_dt   = latest + timedelta(days=ovulation_day - 1)
        possible_ovulation_date = possible_ovulation_dt.strftime("%Y-%m-%d")

        fertile_window = {
            "start": (possible_ovulation_dt - timedelta(days=5)).strftime("%Y-%m-%d"),
            "end":   (possible_ovulation_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
        }

    return {
        "cycle_lengths":          cycle_lengths,
        "estimated_cycle_length": avg_cycle_length,
        "cycle_day":              cycle_day,
        "predicted_next_period":  predicted_next_period,
        "next_period_window":     next_period_window,
        "possible_ovulation_day": ovulation_day,
        "possible_ovulation_date": possible_ovulation_date,
        "fertile_window":         fertile_window,
        "regularity_status":      get_regularity_status(cycle_lengths),
        "forecast_confidence":    get_forecast_confidence(cycle_lengths),
        "phase_probs":            phase_probs_from_cycle_day(cycle_day, avg_cycle_length),
    }
