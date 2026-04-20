"""Repository for cross-session performance analytics queries."""

from dataclasses import dataclass
from datetime import datetime

@dataclass
class SessionPerformance:
    session_number: int
    session_id: int
    started_at: datetime
    error_count: int | None   
    has_error_data: bool
    duration_seconds: float
    avg_rmssd: float | None
    avg_cli: float | None

def get_session_series(db) -> list[SessionPerformance]:
    conn = db.get_connection()
    
    query = """
    SELECT 
        s.id, 
        s.started_at, 
        s.ended_at, 
        s.error_count,
        AVG(h.rmssd) as avg_rmssd,
        AVG(c.cli) as avg_cli
    FROM sessions s
    LEFT JOIN hrv_samples h ON h.session_id = s.id
    LEFT JOIN cli_samples c ON c.session_id = s.id
    WHERE s.ended_at IS NOT NULL
    GROUP BY s.id
    ORDER BY s.started_at ASC
    """
    
    rows = conn.execute(query).fetchall()
    series = []
    
    for i, row in enumerate(rows, start=1):
        started_at = row['started_at']
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
            
        ended_at = row['ended_at']
        if isinstance(ended_at, str):
            ended_at = datetime.fromisoformat(ended_at)
            
        error_count = row['error_count']
        has_error_data = error_count is not None
        
        series.append(SessionPerformance(
            session_number=i,
            session_id=row['id'],
            started_at=started_at,
            error_count=error_count, 
            has_error_data=has_error_data,
            duration_seconds=(ended_at - started_at).total_seconds(),
            avg_rmssd=row['avg_rmssd'],
            avg_cli=row['avg_cli']
        ))
        
    return series