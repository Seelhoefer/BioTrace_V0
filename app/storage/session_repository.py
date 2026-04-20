"""Session repository — CRUD operations for the sessions table.

All database interactions related to sessions are centralised here.
Repositories receive a :class:`~app.storage.database.DatabaseManager`
instance via their constructor (dependency injection — no global state).

Usage::

    from app.storage.database import DatabaseManager
    from app.storage.session_repository import SessionRepository

    db = DatabaseManager()
    repo = SessionRepository(db)
    session_id = repo.create_session("2024-01-15 09:00:00")
"""

import sqlite3
from datetime import datetime
from typing import Optional

from app.storage.database import DatabaseManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SessionRepository:
    """CRUD interface for the ``sessions`` table.

    Args:
        db: Shared database manager providing the SQLite connection.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._conn: sqlite3.Connection = db.get_connection()

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_session(self, started_at: datetime) -> int:
        """Insert a new session row and return its auto-assigned ID.

        Args:
            started_at: The UTC datetime when the session started.

        Returns:
            The ``id`` of the newly created session row.
        """
        cursor = self._conn.execute(
            "INSERT INTO sessions (started_at) VALUES (?)",
            (started_at.isoformat(sep=" "),),
        )
        self._conn.commit()
        session_id = cursor.lastrowid
        logger.info("Session created with id=%d", session_id)
        return session_id

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def end_session(
        self,
        session_id: int,
        ended_at: datetime,
        notes: str = "",
        error_count: Optional[int] = None,
    ) -> None:
        """Record the end time of a session.

        Args:
            session_id: The session to update.
            ended_at: The UTC datetime when the session ended.
            notes: Optional free-text notes from the operator.
            error_count: Optional surgical error count.
        """
        self._conn.execute(
            "UPDATE sessions SET ended_at = ?, notes = ?, error_count = ? WHERE id = ?",
            (ended_at.isoformat(sep=" "), notes, error_count, session_id),
        )
        self._conn.commit()
        logger.info("Session %d ended at %s (errors=%s)", session_id, ended_at, error_count)

    def save_nasa_tlx_score(self, session_id: int, score: float) -> None:
        """Persist the NASA-TLX composite score for a session.

        Args:
            session_id: The session to update.
            score: NASA-TLX weighted workload score (0–100 scale).
        """
        self._conn.execute(
            "UPDATE sessions SET nasa_tlx_score = ? WHERE id = ?",
            (score, session_id),
        )
        self._conn.commit()
        logger.info("NASA-TLX score %.1f saved for session %d", score, session_id)

    def set_session_name(self, session_id: int, name: str) -> None:
        """Update the custom name for a session.

        Args:
            session_id: The session to rename.
            name: Custom session name (e.g. "Trial 1 - Participant A").
        """
        self._conn.execute(
            "UPDATE sessions SET name = ? WHERE id = ?",
            (name, session_id),
        )
        self._conn.commit()
        logger.info("Session %d renamed to '%s'", session_id, name)

    def set_video_path(self, session_id: int, path: str) -> None:
        """Store the video recording file path for a session.

        Args:
            session_id: The session to update.
            path: Absolute filesystem path to the MP4 recording.
        """
        self._conn.execute(
            "UPDATE sessions SET video_path = ? WHERE id = ?",
            (path, session_id),
        )
        self._conn.commit()
        logger.info("Session %d video recording saved at %s", session_id, path)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_video_path(self, session_id: int) -> Optional[str]:
        """Return the video file path for a session, or None if not recorded.

        Args:
            session_id: The session to retrieve.

        Returns:
            The ``video_path`` string or ``None``.
        """
        cursor = self._conn.execute(
            "SELECT video_path FROM sessions WHERE id = ?", (session_id,)
        )
        row = cursor.fetchone()
        return row["video_path"] if row else None

    def get_all_sessions(self) -> list[sqlite3.Row]:
        """Return all sessions ordered newest-first.

        Returns:
            List of :class:`sqlite3.Row` objects with columns matching the
            ``sessions`` table schema.
        """
        cursor = self._conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC"
        )
        return cursor.fetchall()

    def get_completed_sessions(self) -> list[sqlite3.Row]:
        """Return only sessions that have been properly ended, newest-first.

        Excludes sessions where ``ended_at`` is NULL (e.g. abandoned runs
        caused by app crashes).  Used by the sidebar recent-sessions list so
        users are never shown an incomplete session with no exportable data.

        Returns:
            List of :class:`sqlite3.Row` objects ordered by ``started_at`` DESC.
        """
        cursor = self._conn.execute(
            "SELECT * FROM sessions WHERE ended_at IS NOT NULL ORDER BY started_at DESC"
        )
        return cursor.fetchall()

    def get_session(self, session_id: int) -> Optional[sqlite3.Row]:
        """Fetch a single session by ID.

        Args:
            session_id: Primary key of the session to retrieve.

        Returns:
            A :class:`sqlite3.Row` if found, otherwise ``None``.
        """
        cursor = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        return cursor.fetchone()

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_session(self, session_id: int) -> None:
        """Delete a session and all its associated samples (cascade).

        Args:
            session_id: Primary key of the session to remove.
        """
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        logger.info("Session %d deleted (cascade).", session_id)

    def delete_all_sessions(self) -> None:
        """Delete every session and all associated samples (cascade).

        Used by the Settings page "Delete All Data" action.
        """
        self._conn.execute("DELETE FROM sessions")
        self._conn.commit()
        logger.info("All sessions deleted.")
