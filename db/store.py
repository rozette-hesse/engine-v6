"""
Daily Log Store
===============
SQLite-backed persistence for per-user daily observations.

Each row represents one day's signals for one user.  The primary key is
(user_id, log_date) so re-submitting the same day overwrites rather than
duplicates.

Schema
------
Inputs stored
  period_starts        — JSON list of period-start dates the user provided
  period_start_logged  — whether the user flagged a new period today
  symptom_severities   — JSON dict {symptom: 0-5 ordinal} (canonical form)
  symptoms             — JSON list of binary symptom names (derived from severities)
  cervical_mucus       — observed mucus type string
  appetite             — 0-10 appetite level
  exercise_level       — 0-10 exercise level
  flow                 — flow descriptor string

Outputs stored
  final_phase          — predicted phase label
  final_phase_probs    — JSON dict {phase: probability}
  fusion_mode          — which fusion branch ran (fused_non_menstrual, etc.)
  cycle_day            — L1 cycle day estimate
  model_version        — L2 model version used (v4, etc.)

The DB file is controlled by INBALANCE_DB_PATH (defaults to inbalance.db in
the project root).  Swap to Postgres later by replacing _conn() — the public
API is unchanged.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from core.config import settings

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "inbalance.db"
_DB_PATH    = getattr(settings, "db_path", _DEFAULT_DB)


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(_DB_PATH))
    con.row_factory = sqlite3.Row
    return con


# ── Schema management ─────────────────────────────────────────────────────────

_CREATE_DAILY_LOGS = """
    CREATE TABLE IF NOT EXISTS daily_logs (
        user_id              TEXT    NOT NULL,
        log_date             TEXT    NOT NULL,

        -- Cycle timing inputs
        period_starts        TEXT,
        period_start_logged  INTEGER DEFAULT 0,

        -- Symptom inputs (both forms kept for compatibility)
        symptom_severities   TEXT,
        symptoms             TEXT,
        cervical_mucus       TEXT    DEFAULT 'unknown',
        appetite             INTEGER DEFAULT 0,
        exercise_level       INTEGER DEFAULT 0,
        flow                 TEXT    DEFAULT 'none',

        -- Prediction outputs
        final_phase          TEXT,
        final_phase_probs    TEXT,
        fusion_mode          TEXT,
        cycle_day            INTEGER,
        model_version        TEXT,

        -- Metadata
        created_at           TEXT    DEFAULT (datetime('now')),
        updated_at           TEXT    DEFAULT (datetime('now')),

        PRIMARY KEY (user_id, log_date)
    )
"""

# Columns added after initial release — migrate_db() adds these to existing DBs.
_NEW_COLUMNS: List[tuple] = [
    ("period_starts",       "TEXT"),
    ("period_start_logged", "INTEGER DEFAULT 0"),
    ("symptom_severities",  "TEXT"),
    ("final_phase",         "TEXT"),
    ("final_phase_probs",   "TEXT"),
    ("fusion_mode",         "TEXT"),
    ("cycle_day",           "INTEGER"),
    ("model_version",       "TEXT"),
    # SQLite's ALTER TABLE ADD COLUMN requires a constant default, so we
    # can't reuse the `(datetime('now'))` default from _CREATE_DAILY_LOGS.
    # save_log() writes this column explicitly, so NULL-by-default is fine.
    ("updated_at",          "TEXT"),
]


def init_db() -> None:
    """Create tables if they don't exist. Called once at app startup."""
    with _conn() as con:
        con.execute(_CREATE_DAILY_LOGS)


def migrate_db() -> None:
    """
    Add any columns that exist in the current schema but not in an older DB.
    Safe to call on every startup — ALTER TABLE is skipped if the column
    already exists (SQLite raises OperationalError for duplicate columns).
    """
    with _conn() as con:
        for col_name, col_def in _NEW_COLUMNS:
            try:
                con.execute(f"ALTER TABLE daily_logs ADD COLUMN {col_name} {col_def}")
            except sqlite3.OperationalError:
                pass  # column already present


# ── Write ─────────────────────────────────────────────────────────────────────

def save_log(
    user_id: str,
    log_date: str,
    cycle_inputs: Dict,
    *,
    period_starts: Optional[List[str]] = None,
    period_start_logged: bool = False,
    symptom_severities: Optional[Dict[str, int]] = None,
    fusion_output: Optional[Dict] = None,
) -> None:
    """
    Persist one day's inputs and prediction outputs for a user.

    Parameters
    ----------
    user_id              : stable user identifier from the auth provider
    log_date             : YYYY-MM-DD date this log belongs to
    cycle_inputs         : dict from build_cycle_inputs() — symptoms, mucus, etc.
    period_starts        : list of period-start dates passed to the cycle engine
    period_start_logged  : True if user flagged a new period today
    symptom_severities   : {symptom: 0-5} ordinal dict — preferred symptom form
    fusion_output        : full dict returned by get_fused_output()
    """
    fo = fusion_output or {}
    l1 = fo.get("layer1") or {}

    with _conn() as con:
        con.execute("""
            INSERT OR REPLACE INTO daily_logs (
                user_id, log_date,
                period_starts, period_start_logged,
                symptom_severities, symptoms,
                cervical_mucus, appetite, exercise_level, flow,
                final_phase, final_phase_probs,
                fusion_mode, cycle_day, model_version,
                updated_at
            ) VALUES (
                ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                datetime('now')
            )
        """, (
            user_id,
            log_date,
            # inputs
            json.dumps(period_starts or []),
            int(period_start_logged),
            json.dumps(symptom_severities or {}),
            json.dumps(cycle_inputs.get("symptoms", [])),
            cycle_inputs.get("cervical_mucus", "unknown"),
            int(cycle_inputs.get("appetite", 0)),
            int(cycle_inputs.get("exerciselevel", 0)),
            cycle_inputs.get("flow", "none"),
            # outputs
            fo.get("final_phase"),
            json.dumps(fo.get("final_phase_probs") or {}),
            fo.get("mode"),
            l1.get("cycle_day"),
            fo.get("model_version"),
        ))


# ── Read ──────────────────────────────────────────────────────────────────────

def get_recent_logs(
    user_id: str,
    before_date: str,
    n: int = 3,
    max_gap_days: int = 4,
) -> List[Dict]:
    """
    Return up to n daily logs strictly before before_date, oldest-first.

    Only logs within ``max_gap_days`` of before_date are returned.  Logs
    older than that are stale — passing them as recent context to the cycle
    engine would mislead the feature builder into treating a week-old symptom
    report as "yesterday".

    Parameters
    ----------
    user_id      : user identifier
    before_date  : upper bound date (YYYY-MM-DD), exclusive
    n            : maximum number of logs to return (default 3)
    max_gap_days : oldest log accepted, relative to before_date (default 4)

    The result is ready to pass as recent_daily_logs to get_fused_output().
    Includes symptom_severities so the v4/v5 feature builders receive ordinal
    values rather than falling back to binary-to-ordinal conversion.
    """
    with _conn() as con:
        rows = con.execute("""
            SELECT log_date, symptoms, symptom_severities,
                   cervical_mucus, appetite, exercise_level, flow
            FROM daily_logs
            WHERE user_id = ?
              AND log_date < ?
              AND log_date >= date(?, ?)
            ORDER BY log_date DESC
            LIMIT ?
        """, (
            user_id,
            before_date,
            before_date, f"-{max_gap_days} days",
            n,
        )).fetchall()

    result = []
    for row in reversed(rows):
        sev_raw = row["symptom_severities"]
        result.append({
            "date":               row["log_date"],
            "symptoms":           json.loads(row["symptoms"] or "[]"),
            "symptom_severities": json.loads(sev_raw) if sev_raw else {},
            "cervical_mucus":     row["cervical_mucus"],
            "exerciselevel":      row["exercise_level"],
            "appetite":           row["appetite"],
            "flow":               row["flow"],
        })
    return result


def get_history_logs(user_id: str, n: int = 180) -> List[Dict]:
    """
    Return up to n historical daily logs for a user, oldest-first.

    Intended for the personalization layer — includes cycle_day, final_phase,
    symptom_severities, and log_date so per-phase symptom baselines can be
    computed from the user's own history.

    Parameters
    ----------
    user_id : user identifier
    n       : maximum number of days to return (default 180 ≈ 6 months)
    """
    with _conn() as con:
        rows = con.execute("""
            SELECT log_date, symptom_severities, cervical_mucus,
                   appetite, exercise_level, flow,
                   final_phase, cycle_day, period_starts
            FROM daily_logs
            WHERE user_id = ?
            ORDER BY log_date ASC
            LIMIT ?
        """, (user_id, n)).fetchall()

    result = []
    for row in rows:
        sev_raw    = row["symptom_severities"]
        starts_raw = row["period_starts"]
        result.append({
            "date":               row["log_date"],
            "symptom_severities": json.loads(sev_raw)    if sev_raw    else {},
            "cervical_mucus":     row["cervical_mucus"],
            "appetite":           row["appetite"],
            "exerciselevel":      row["exercise_level"],
            "flow":               row["flow"],
            "final_phase":        row["final_phase"],
            "cycle_day":          row["cycle_day"],
            "period_starts":      json.loads(starts_raw) if starts_raw else [],
        })
    return result
