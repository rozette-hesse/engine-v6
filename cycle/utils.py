"""Shared date/probability utilities for the cycle engine."""

from datetime import datetime


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def days_between(d1: datetime, d2: datetime) -> int:
    return (d2 - d1).days


def normalize_probs(probs: dict) -> dict:
    """Normalize a probability dict so values sum to 1. Handles zero-sum gracefully."""
    total = sum(probs.values())
    if total <= 0:
        n = len(probs)
        return {k: 1.0 / n for k in probs}
    return {k: float(v) / total for k, v in probs.items()}
