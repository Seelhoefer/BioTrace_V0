"""Integration tests for PerformanceRepository."""

import os
from datetime import datetime, timedelta
import pytest
from app.storage.database import DatabaseManager
from app.analytics.performance_repository import get_session_series

@pytest.fixture
def db():
    """Provide a fresh SQLite database for each test."""
    test_db_path = "test_analytics.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    db_manager = DatabaseManager(test_db_path)
    yield db_manager
    
    db_manager.close()
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

def test_get_session_series_ordering(db):
    """get_session_series returns completed sessions ordered by started_at."""
    conn = db.get_connection()
    t1 = datetime(2026, 1, 1, 10, 0, 0)
    t2 = datetime(2026, 1, 2, 10, 0, 0)
    t3 = datetime(2026, 1, 3, 10, 0, 0)
    
    # Insert sessions out of order
    # t3 (completed)
    conn.execute("INSERT INTO sessions (started_at, ended_at) VALUES (?, ?)", 
                 (t3.isoformat(sep=" "), (t3 + timedelta(minutes=10)).isoformat(sep=" ")))
    # t1 (completed)
    conn.execute("INSERT INTO sessions (started_at, ended_at) VALUES (?, ?)", 
                 (t1.isoformat(sep=" "), (t1 + timedelta(minutes=10)).isoformat(sep=" ")))
    # t2 (incomplete)
    conn.execute("INSERT INTO sessions (started_at) VALUES (?)", 
                 (t2.isoformat(sep=" "),))
    conn.commit()
    
    series = get_session_series(db)
    
    # t2 should be excluded. t1 and t3 should be present, t1 first.
    assert len(series) == 2
    assert series[0].started_at == t1
    assert series[0].session_number == 1
    assert series[1].started_at == t3
    assert series[1].session_number == 2

def test_get_session_series_error_data_handling(db):
    """get_session_series handles NULL error_count correctly."""
    conn = db.get_connection()
    t1 = datetime(2026, 1, 1, 10, 0, 0)
    t2 = datetime(2026, 1, 2, 10, 0, 0)
    
    # t1 with error_count=5
    conn.execute("INSERT INTO sessions (started_at, ended_at, error_count) VALUES (?, ?, ?)", 
                 (t1.isoformat(sep=" "), (t1 + timedelta(minutes=10)).isoformat(sep=" "), 5))
    # t2 with NULL error_count
    conn.execute("INSERT INTO sessions (started_at, ended_at, error_count) VALUES (?, ?, ?)", 
                 (t2.isoformat(sep=" "), (t2 + timedelta(minutes=10)).isoformat(sep=" "), None))
    conn.commit()
    
    series = get_session_series(db)
    
    assert len(series) == 2
    assert series[0].error_count == 5
    assert series[0].has_error_data is True
    assert series[1].error_count == 0
    assert series[1].has_error_data is False

def test_get_session_series_averages(db):
    """get_session_series computes averages for RMSSD and CLI."""
    conn = db.get_connection()
    t1 = datetime(2026, 1, 1, 10, 0, 0)
    conn.execute("INSERT INTO sessions (id, started_at, ended_at) VALUES (1, ?, ?)", 
                 (t1.isoformat(sep=" "), (t1 + timedelta(minutes=10)).isoformat(sep=" ")))
    
    # hrv_samples: 30, 40, 50 -> avg 40
    conn.execute("INSERT INTO hrv_samples (session_id, timestamp, rr_interval, rmssd) VALUES (1, 1, 800, 30.0)")
    conn.execute("INSERT INTO hrv_samples (session_id, timestamp, rr_interval, rmssd) VALUES (1, 2, 800, 40.0)")
    conn.execute("INSERT INTO hrv_samples (session_id, timestamp, rr_interval, rmssd) VALUES (1, 3, 800, 50.0)")
    
    # cli_samples: 0.2, 0.4 -> avg 0.3
    conn.execute("INSERT INTO cli_samples (session_id, timestamp, cli) VALUES (1, 1, 0.2)")
    conn.execute("INSERT INTO cli_samples (session_id, timestamp, cli) VALUES (1, 2, 0.4)")
    conn.commit()
    
    series = get_session_series(db)
    
    assert len(series) == 1
    assert series[0].avg_rmssd == pytest.approx(40.0)
    assert series[0].avg_cli == pytest.approx(0.3)

def test_get_session_series_has_error_data_filtering(db):
    """get_session_series correctly flags sessions with and without error data."""
    conn = db.get_connection()
    t1 = datetime(2026, 1, 1, 10, 0, 0)
    t2 = datetime(2026, 1, 2, 10, 0, 0)
    
    # Session with errors
    conn.execute("INSERT INTO sessions (started_at, ended_at, error_count) VALUES (?, ?, ?)", 
                 (t1.isoformat(sep=" "), (t1 + timedelta(minutes=10)).isoformat(sep=" "), 0))
    # Session with NULL errors
    conn.execute("INSERT INTO sessions (started_at, ended_at, error_count) VALUES (?, ?, ?)", 
                 (t2.isoformat(sep=" "), (t2 + timedelta(minutes=10)).isoformat(sep=" "), None))
    conn.commit()
    
    series = get_session_series(db)
    assert len(series) == 2
    assert series[0].has_error_data is True
    assert series[1].has_error_data is False
