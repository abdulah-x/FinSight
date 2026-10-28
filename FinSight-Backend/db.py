"""Postgres storage for predictions, so each bias can be graded later (Stage 7).

Every new analysis is stored with its entry price, ATR and the time it should be
checked. If DATABASE_URL isn't set, storage is off and the app still runs.
All SQL uses parameters (%s), never string formatting, so values can't inject SQL.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import psycopg
import psycopg.rows

HORIZON = {"1H": timedelta(hours=1), "4H": timedelta(hours=4), "1D": timedelta(days=1), "1W": timedelta(weeks=1)}

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id           BIGSERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    asset        TEXT NOT NULL,
    timeframe    TEXT NOT NULL,
    bias         TEXT NOT NULL,
    score        INTEGER NOT NULL,
    bucket       TEXT NOT NULL,
    net          DOUBLE PRECISION NOT NULL,
    coverage     DOUBLE PRECISION NOT NULL,
    trend        TEXT,                      -- price trend label, for the "follow the EMA" baseline
    entry_price  DOUBLE PRECISION NOT NULL,
    atr          DOUBLE PRECISION NOT NULL,
    resolve_at   TIMESTAMPTZ NOT NULL,
    exit_price   DOUBLE PRECISION,          -- filled in by the resolver (Stage 7)
    correct      BOOLEAN,
    evidence     JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_unresolved ON predictions (resolve_at) WHERE exit_price IS NULL;
-- added in Stage 7: who made the prediction ('agent' = a trader's request, 'collector' = scheduled engine run)
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'agent';
"""


def _url() -> str | None:
    return os.environ.get("DATABASE_URL")


def enabled() -> bool:
    return bool(_url())


def init_db() -> None:
    if not enabled():
        return
    with psycopg.connect(_url()) as conn:
        conn.execute(SCHEMA)


def save_prediction(analysis: dict, source: str = "agent") -> int | None:
    """Store one analysis. Returns the new row id, or None if storage is off or
    the analysis has no price to grade against."""
    price = analysis.get("price")
    if not enabled() or not price:
        return None
    resolve_at = datetime.now(timezone.utc) + HORIZON[analysis["timeframe"]]
    with psycopg.connect(_url()) as conn:
        row = conn.execute(
            """INSERT INTO predictions
                   (asset, timeframe, bias, score, bucket, net, coverage, trend,
                    entry_price, atr, resolve_at, evidence, source)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (analysis["asset"], analysis["timeframe"], analysis["bias"], analysis["score"],
             analysis["bucket"], analysis["net"], analysis["coverage"], price.get("trend"),
             price["last_price"], price["atr14"], resolve_at, json.dumps(analysis["evidence"]), source),
        ).fetchone()
    return row[0]


def due_predictions(limit: int = 500) -> list[dict]:
    """Ungraded predictions whose check time has passed, oldest first."""
    if not enabled():
        return []
    with psycopg.connect(_url()) as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        return cur.execute(
            """SELECT id, asset, bias, entry_price, atr, resolve_at FROM predictions
               WHERE exit_price IS NULL AND resolve_at <= now()
               ORDER BY resolve_at LIMIT %s""",
            (limit,),
        ).fetchall()


def record_outcome(prediction_id: int, exit_price: float, correct: bool) -> None:
    with psycopg.connect(_url()) as conn:
        conn.execute("UPDATE predictions SET exit_price = %s, correct = %s WHERE id = %s",
                     (exit_price, correct, prediction_id))


def resolved_predictions() -> list[dict]:
    """Every graded prediction, for the calibration report."""
    with psycopg.connect(_url()) as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        return cur.execute(
            """SELECT id, created_at, source, asset, timeframe, bias, score, bucket, trend,
                      entry_price, exit_price, atr, correct
               FROM predictions WHERE exit_price IS NOT NULL ORDER BY created_at"""
        ).fetchall()
