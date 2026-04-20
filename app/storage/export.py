"""CSV, JSON, and Excel export utilities for BioTrace session data.

Usage::

    from app.storage.export import SessionExporter
    exporter = SessionExporter(db)
    exporter.export_csv(session_id=1, path="session_1.csv")
    exporter.export_json(session_id=1, path="session_1.json")
    exporter.export_excel(session_id=1, path="session_1.xlsx")
"""

import csv
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from app.storage.database import DatabaseManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SessionExporter:
    """Exports session data to CSV or JSON files.

    Args:
        db: Shared database manager providing the SQLite connection.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._conn: sqlite3.Connection = db.get_connection()

    @staticmethod
    def _parse_datetime(raw: object) -> datetime | None:
        """Parse an ISO timestamp returned by SQLite."""
        if raw in (None, ""):
            return None
        try:
            return datetime.fromisoformat(str(raw))
        except ValueError:
            return None

    @staticmethod
    def _format_datetime(raw: object) -> str | None:
        """Format an ISO timestamp as ``YYYY-MM-DD, HH:MM:SS``."""
        dt = SessionExporter._parse_datetime(raw)
        if dt is None:
            return None
        return dt.strftime("%Y-%m-%d, %H:%M:%S")

    @staticmethod
    def _format_elapsed_timestamp(started_at: datetime | None, elapsed_s: object) -> str | None:
        """Convert an elapsed-session timestamp to a formatted absolute time."""
        if started_at is None or elapsed_s is None:
            return None
        try:
            elapsed = float(elapsed_s)
        except (TypeError, ValueError):
            return None
        return (started_at + timedelta(seconds=elapsed)).strftime("%Y-%m-%d, %H:%M:%S")

    @staticmethod
    def _mean_or_none(values: list[object]) -> float | None:
        """Return the arithmetic mean for numeric values, or None."""
        filtered = [float(value) for value in values if value is not None]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    def _fetch_session_data(self, session_id: int) -> dict:
        """Collect all data rows for a session into a dict.

        Args:
            session_id: The session to export.

        Returns:
            Dictionary with keys ``session``, ``hrv``, ``pupil``, ``cli``.
        """
        session = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()

        hrv = self._conn.execute(
            "SELECT timestamp, rr_interval, bpm, rmssd, delta_rmssd FROM hrv_samples "
            "WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()

        pupil = self._conn.execute(
            "SELECT timestamp, left_diameter, right_diameter, pdi "
            "FROM pupil_samples WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()

        cli = self._conn.execute(
            "SELECT timestamp, cli FROM cli_samples "
            "WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()

        return {
            "session": dict(session) if session else {},
            "hrv": [dict(r) for r in hrv],
            "pupil": [dict(r) for r in pupil],
            "cli": [dict(r) for r in cli],
        }

    def export_csv(self, session_id: int, path: str | Path) -> None:
        """Export session samples to a flat CSV file.

        The CSV is keyed on ``timestamp`` and merges HRV, pupil, and CLI
        columns side by side, leaving blanks where data is sparse.

        Args:
            session_id: The session to export.
            path: Destination file path (created or overwritten).
        """
        data = self._fetch_session_data(session_id)
        path = Path(path)

        # Build a unified dict keyed by timestamp.
        combined: dict[float, dict] = {}

        for row in data["hrv"]:
            t = row["timestamp"]
            combined.setdefault(t, {})["rr_interval"] = row["rr_interval"]
            combined[t]["rmssd"] = row["rmssd"]

        for row in data["pupil"]:
            t = row["timestamp"]
            combined.setdefault(t, {})["left_diameter"] = row["left_diameter"]
            combined[t]["right_diameter"] = row["right_diameter"]
            combined[t]["pdi"] = row["pdi"]

        for row in data["cli"]:
            t = row["timestamp"]
            combined.setdefault(t, {})["cli"] = row["cli"]

        fieldnames = ["timestamp", "rr_interval", "rmssd",
                      "left_diameter", "right_diameter", "pdi", "cli"]

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for ts in sorted(combined):
                row = {"timestamp": ts}
                row.update(combined[ts])
                writer.writerow(row)

        logger.info("Session %d exported to CSV: %s", session_id, path)

    def export_json(self, session_id: int, path: str | Path) -> None:
        """Export all session data to a structured JSON file.

        Args:
            session_id: The session to export.
            path: Destination file path (created or overwritten).
        """
        data = self._fetch_session_data(session_id)
        path = Path(path)

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Session %d exported to JSON: %s", session_id, path)

    def export_excel(self, session_id: int, path: str | Path) -> None:
        """Export all session data to a multi-sheet Excel (.xlsx) file.

        Sheets produced:
                - **Session Info** — session metadata (start/end time, duration, notes,
                    average HRV, average pupil dilation change).
        - **HRV** — per-beat data: timestamp, RR interval, BPM, RMSSD,
          delta-RMSSD.
        - **Pupil** — per-sample data: timestamp, left diameter, right diameter,
          pupil dilation index.
        - **CLI** — per-sample data: timestamp, Cognitive Load Index.

        Args:
            session_id: The session to export.
            path: Destination ``.xlsx`` file path (created or overwritten).
        """
        data = self._fetch_session_data(session_id)
        path = Path(path)

        # ── Sheet: Session Info ────────────────────────────────────────────
        session = data["session"]
        started_dt = self._parse_datetime(session.get("started_at"))

        # Compute duration in whole seconds (None when session not yet ended).
        duration_s: int | None = None
        started_raw = session.get("started_at")
        ended_raw   = session.get("ended_at")
        ended_dt = self._parse_datetime(ended_raw)
        if started_dt and ended_dt:
            duration_s = int((ended_dt - started_dt).total_seconds())
        elif started_raw and ended_raw:
            logger.warning("Could not parse duration for session %s", session_id)

        hrv_count = len(data["hrv"])
        avg_rmssd = self._mean_or_none([row.get("rmssd") for row in data["hrv"]])
        avg_pdi = self._mean_or_none([row.get("pdi") for row in data["pupil"]])

        info_df = pd.DataFrame(
            [
                ("Session ID",     session.get("id")),
                ("Started at",     self._format_datetime(session.get("started_at"))),
                ("Ended at",       self._format_datetime(session.get("ended_at"))),
                ("Duration (s)",   duration_s),
                ("HRV samples",    hrv_count),
                ("Average HRV (RMSSD)", avg_rmssd),
                ("Average Pupil Dilation Change (PDI)", avg_pdi),
                ("Notes",          session.get("notes", "")),
            ],
            columns=["Field", "Value"],
        )

        # ── Sheet: Measurements (merged HRV + Pupil, one row per timestamp) ──
        measurements_df = self._measurements_df(data, started_dt)

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            info_df.to_excel(writer,          sheet_name="Session Info",  index=False)
            measurements_df.to_excel(writer,  sheet_name="Measurements",  index=False)

        logger.info("Session %d exported to Excel: %s", session_id, path)

    def _measurements_df(self, data: dict, started_dt: datetime | None) -> pd.DataFrame:
        """Build the Measurements DataFrame from a session data dict.

        Merges HRV and pupil samples on timestamp (outer join), renames
        columns to human-readable labels, and returns NaN as None.

        Args:
            data: Dict returned by ``_fetch_session_data``.

        Returns:
            DataFrame with columns: Time, BPM, HRV, RMSSD, Delta RMSSD,
            Pupil Diameter [px], Delta Pupil Dilation [PDI].
        """
        hrv_df = pd.DataFrame(
            data["hrv"],
            columns=["timestamp", "rr_interval", "bpm", "rmssd", "delta_rmssd"],
        ) if data["hrv"] else pd.DataFrame(
            columns=["timestamp", "rr_interval", "bpm", "rmssd", "delta_rmssd"]
        )

        pupil_df = pd.DataFrame(
            data["pupil"],
            columns=["timestamp", "left_diameter", "right_diameter", "pdi"],
        ) if data["pupil"] else pd.DataFrame(
            columns=["timestamp", "left_diameter", "right_diameter", "pdi"]
        )

        if not pupil_df.empty:
            pupil_df["pupil_diameter"] = (
                pupil_df[["left_diameter", "right_diameter"]].mean(axis=1)
            )
        else:
            pupil_df["pupil_diameter"] = pd.Series(dtype=float)

        merged = pd.merge(
            hrv_df[["timestamp", "bpm", "rr_interval", "rmssd", "delta_rmssd"]],
            pupil_df[["timestamp", "pupil_diameter", "pdi"]],
            on="timestamp",
            how="outer",
        ).sort_values("timestamp").reset_index(drop=True)

        merged = merged.where(pd.notna(merged), other=None)

        if started_dt is not None:
            merged["timestamp"] = merged["timestamp"].apply(
                lambda elapsed: self._format_elapsed_timestamp(started_dt, elapsed)
            )
        else:
            merged["timestamp"] = merged["timestamp"].apply(
                lambda elapsed: str(elapsed) if elapsed is not None else None
            )

        return merged.rename(columns={
            "timestamp":        "Time",
            "bpm":              "BPM",
            "rr_interval":      "HRV",
            "rmssd":            "RMSSD",
            "delta_rmssd":      "Delta RMSSD",
            "pupil_diameter":   "Pupil Diameter [px]",
            "pdi":              "Delta Pupil Dilation [PDI]",
        })[["Time", "BPM", "HRV", "RMSSD", "Delta RMSSD",
            "Pupil Diameter [px]", "Delta Pupil Dilation [PDI]"]]

    def export_all_sessions(self, path: str | Path) -> None:
        """Export all completed sessions to a single multi-sheet Excel file.

        Sheets produced:
        - **Summary** — one row per session: ID, date, duration, NASA-TLX score,
          error count, notes.
        - **Session {id}** — per-session measurements for each completed session
          (same columns as the Measurements sheet in ``export_excel``).

        If no completed sessions exist, writes a Summary sheet with a single
        informational row.

        Args:
            path: Destination ``.xlsx`` file path (created or overwritten).
        """
        path = Path(path)
        sessions = self._conn.execute(
            "SELECT * FROM sessions WHERE ended_at IS NOT NULL ORDER BY started_at DESC"
        ).fetchall()

        summary_rows = []
        for s in sessions:
            try:
                duration_s: int | None = int(
                    (datetime.fromisoformat(str(s["ended_at"])) -
                     datetime.fromisoformat(str(s["started_at"]))).total_seconds()
                )
            except ValueError:
                logger.warning("Could not parse duration for session %s", s["id"])
                duration_s = None

            summary_rows.append({
                "Session ID":    s["id"],
                "Date":          self._format_datetime(s["started_at"]),
                "Duration (s)":  duration_s,
                "NASA-TLX Score": s["nasa_tlx_score"],
                "Error Count":   s["error_count"],
                "Notes":         s["notes"] or "",
            })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
        else:
            summary_df = pd.DataFrame([{"Session ID": "No sessions recorded"}])

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            for s in sessions:
                sid = s["id"]
                sheet_name = f"Session {sid}"
                data = self._fetch_session_data(sid)
                mdf = self._measurements_df(data, self._parse_datetime(s["started_at"]))
                mdf.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info("All sessions exported to Excel: %s", path)
