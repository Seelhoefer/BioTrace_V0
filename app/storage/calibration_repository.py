"""Calibration repository — CRUD for the calibrations table.

Stores and retrieves pre-session baseline measurements (resting RMSSD and
pupil diameter) recorded during the calibration wizard.

Usage::

    repo = CalibrationRepository(db)
    cal_id = repo.save_calibration(
        session_id=1,
        baseline_rmssd=38.4,
        baseline_pupil_px=120.0,
        duration_seconds=60,
    )
"""

import sqlite3
from datetime import datetime, timezone
from typing import Optional

from app.storage.database import DatabaseManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CalibrationRepository:
    """CRUD interface for the ``calibrations`` table.

    Args:
        db: Shared database manager providing the SQLite connection.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._conn: sqlite3.Connection = db.get_connection()

    def save_calibration(
        self,
        session_id: int,
        baseline_rmssd: float,
        baseline_pupil_px: float,
        duration_seconds: int,
        baseline_rmssd_std: float = 0.0,
        baseline_pupil_std: float = 0.0,
    ) -> int:
        """Insert a calibration baseline record and return its ID.

        Args:
            session_id: The session this calibration belongs to.
            baseline_rmssd: Resting RMSSD in milliseconds.
            baseline_pupil_px: Resting pupil diameter in px.
            duration_seconds: Actual recording duration in seconds.
            baseline_rmssd_std: Standard deviation of RMSSD during calibration.
            baseline_pupil_std: Standard deviation of pupil diameter during calibration.

        Returns:
            The ``id`` of the new calibration row.
        """
        recorded_at = datetime.now(tz=timezone.utc).isoformat(sep=" ")
        cursor = self._conn.execute(
            """
            INSERT INTO calibrations
                (
                    session_id,
                    recorded_at,
                    duration_seconds,
                    baseline_rmssd,
                    baseline_rmssd_std,
                    baseline_pupil_mm,
                    baseline_pupil_std
                )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                recorded_at,
                duration_seconds,
                baseline_rmssd,
                baseline_rmssd_std,
                baseline_pupil_px,
                baseline_pupil_std,
            ),
        )
        self._conn.commit()
        cal_id = cursor.lastrowid
        logger.info(
            "Calibration saved: id=%d session=%d rmssd=%.2f±%.2f pupil=%.3f±%.3f px",
            cal_id,
            session_id,
            baseline_rmssd,
            baseline_rmssd_std,
            baseline_pupil_px,
            baseline_pupil_std,
        )
        return cal_id

    def get_latest_for_session(self, session_id: int) -> Optional[sqlite3.Row]:
        """Fetch the most recent calibration record for a session.

        Args:
            session_id: The session to query.

        Returns:
            A :class:`sqlite3.Row` if found, otherwise ``None``.
        """
        cursor = self._conn.execute(
            "SELECT * FROM calibrations WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        )
        return cursor.fetchone()

    def save_hrv_samples_bulk(
        self,
        session_id: int,
        samples: list[tuple[float, float, float | None, float | None, float | None]],
    ) -> None:
        """Bulk-insert HRV samples accumulated during a session.

        Args:
            session_id: The owning session ID.
            samples: List of ``(timestamp, rr_interval_ms, rmssd, bpm, delta_rmssd)``
                     tuples.  Any of the last three values may be ``None``.
        """
        self._conn.executemany(
            """
            INSERT INTO hrv_samples
                (session_id, timestamp, rr_interval, rmssd, bpm, delta_rmssd)
            VALUES (?,?,?,?,?,?)
            """,
            [(session_id, ts, rr, rmssd, bpm, delta) for ts, rr, rmssd, bpm, delta in samples],
        )
        self._conn.commit()
        logger.info("Saved %d HRV samples for session %d.", len(samples), session_id)

    def save_pupil_samples_bulk(
        self,
        session_id: int,
        samples: list[tuple[float, float | None, float | None, float | None]],
    ) -> None:
        """Bulk-insert pupil samples accumulated during a session.

        Args:
            session_id: The owning session ID.
            samples: List of ``(timestamp, left_px, right_px, pdi_or_None)`` tuples.
        """
        self._conn.executemany(
            """
            INSERT INTO pupil_samples (session_id, timestamp, left_diameter, right_diameter, pdi)
            VALUES (?,?,?,?,?)
            """,
            [(session_id, ts, l, r, pdi) for ts, l, r, pdi in samples],
        )
        self._conn.commit()
        logger.info("Saved %d pupil samples for session %d.", len(samples), session_id)

    def save_cli_samples_bulk(
        self, session_id: int, samples: list[tuple[float, float]]
    ) -> None:
        """Bulk-insert CLI samples accumulated during a session.

        Args:
            session_id: The owning session ID.
            samples: List of ``(timestamp, cli)`` tuples.
        """
        self._conn.executemany(
            "INSERT INTO cli_samples (session_id, timestamp, cli) VALUES (?,?,?)",
            [(session_id, ts, cli) for ts, cli in samples],
        )
        self._conn.commit()
        logger.info("Saved %d CLI samples for session %d.", len(samples), session_id)
