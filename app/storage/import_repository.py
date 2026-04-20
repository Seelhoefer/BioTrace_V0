"""ImportRepository — persist and retrieve external LapSim datasets.

Allows parsed Excel data to be stored in the local SQLite database so users
can revisit previously analysed learning curves without re-uploading files.
"""

import sqlite3
from typing import Any, Dict, List

from app.analytics.lapsim_parser import ParsedDataset
from app.storage.database import DatabaseManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ImportRepository:
    """CRUD interface for ``imported_datasets`` and ``imported_trials`` tables.

    Args:
        db: Shared database manager providing the SQLite connection.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._conn: sqlite3.Connection = db.get_connection()

    def save_dataset(self, dataset: ParsedDataset, metric_used: str) -> int:
        """Persist a parsed dataset and return its new ID.

        Args:
            dataset: The validated dataset from LapSimParser.
            metric_used: The label of the metric chosen for the primary fit
                         (e.g., "Total Time (s)").
        """
        # 1. Insert dataset metadata
        cursor = self._conn.execute(
            """
            INSERT INTO imported_datasets 
                (filename, participant, exercise, trial_count, metric_used)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                dataset.source_file,
                dataset.participant,
                dataset.exercise,
                len(dataset.trials),
                metric_used
            )
        )
        dataset_id = cursor.lastrowid

        # 2. Prepare trial rows
        trial_rows = []
        for t in dataset.trials:
            # Determine which value was the 'primary' raw_value at import time.
            # This is largely for backwards compatibility or quick-load;
            # the raw columns are the source of truth.
            raw_val = 0.0
            if metric_used == "Total Time (s)":
                raw_val = t.total_time_s or 0.0
            elif metric_used == "Score":
                raw_val = t.score or 0.0
            elif metric_used == "Tissue Damage (#)":
                raw_val = float(t.tissue_damage or 0)

            trial_rows.append((
                dataset_id,
                t.trial_number,
                t.start_time,
                raw_val,
                t.score,
                t.total_time_s,
                t.tissue_damage
            ))

        # 3. Bulk insert trials
        self._conn.executemany(
            """
            INSERT INTO imported_trials
                (dataset_id, trial_number, start_time, raw_value, score, total_time_s, tissue_damage)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            trial_rows
        )
        self._conn.commit()
        logger.info("Dataset %d saved with %d trials.", dataset_id, len(dataset.trials))
        return dataset_id

    def get_all_datasets(self) -> List[Dict[str, Any]]:
        """Return summary of all imported datasets, newest first."""
        cursor = self._conn.execute(
            """
            SELECT id, participant, exercise, trial_count, imported_at, filename, metric_used 
            FROM imported_datasets 
            ORDER BY imported_at DESC
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_trials(self, dataset_id: int) -> List[Dict[str, Any]]:
        """Return all trial rows for a dataset, ordered by chronological sequence."""
        cursor = self._conn.execute(
            "SELECT * FROM imported_trials WHERE dataset_id = ? ORDER BY trial_number",
            (dataset_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete_dataset(self, dataset_id: int) -> None:
        """Remove a dataset and its associated trials from the database."""
        # Foreign key with ON DELETE CASCADE handles the imported_trials.
        self._conn.execute("DELETE FROM imported_datasets WHERE id = ?", (dataset_id,))
        self._conn.commit()
        logger.info("Dataset %d deleted.", dataset_id)
