"""Unit tests for LapSimParser.

Verifies header detection, validation rules, and chronological sorting.
"""

import pytest
import pandas as pd
from pathlib import Path
from app.analytics.lapsim_parser import LapSimParser, LC_MIN_SESSIONS

@pytest.fixture
def parser():
    return LapSimParser()

@pytest.fixture
def dummy_lapsim_excel(tmp_path):
    """Create a valid LapSim-style Excel file with metadata rows."""
    path = tmp_path / "test_lapsim.xlsx"
    
    # Metadata rows
    meta = pd.DataFrame([["Course:", "Advanced Lap"], ["Date:", "2024-01-01"]])
    
    # Header and data
    data = pd.DataFrame([
        ["user1", "2024-01-01 10:00:00", "Grasping", 60.5, 85.0, 2, "Pass"],
        ["user1", "2024-01-01 10:05:00", "Grasping", 55.2, 88.0, 1, "Pass"],
        ["user1", "2024-01-01 09:55:00", "Grasping", 70.0, 80.0, 3, "Pass"], # Out of order
    ], columns=["Login", "Start Time", "Task Name", "Total Time (s)", "Score", "Tissue Damage (#)", "Status"])
    
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        # Write metadata then data below it
        meta.to_excel(writer, sheet_name="Grasping", index=False, header=False)
        data.to_excel(writer, sheet_name="Grasping", index=False, startrow=5)
        
    return path

class TestLapSimParser:
    def test_list_sheets(self, parser, dummy_lapsim_excel):
        sheets = parser.list_sheets(str(dummy_lapsim_excel))
        assert "Grasping" in sheets

    def test_parse_valid_sheet(self, parser, dummy_lapsim_excel):
        parsed = parser.parse(str(dummy_lapsim_excel), "Grasping")
        assert parsed.participant == "user1"
        assert parsed.exercise == "Grasping"
        assert len(parsed.trials) == 3
        # Check sorting (9:55 should be trial 1)
        assert "09:55:00" in parsed.trials[0].start_time
        assert parsed.trials[0].trial_number == 1
        assert parsed.trials[2].trial_number == 3

    def test_parse_multi_participant_raises(self, parser, tmp_path):
        path = tmp_path / "multi_user.xlsx"
        df = pd.DataFrame([
            ["user1", "2024-01-01 10:00", "Task1"],
            ["user2", "2024-01-01 10:05", "Task1"]
        ], columns=["Login", "Start Time", "Task Name"])
        df.to_excel(path, index=False)
        
        with pytest.raises(ValueError, match="Multiple participants found"):
            parser.parse(str(path), "Sheet1")

    def test_parse_multi_task_raises(self, parser, tmp_path):
        path = tmp_path / "multi_task.xlsx"
        df = pd.DataFrame([
            ["user1", "2024-01-01 10:00", "Task1"],
            ["user1", "2024-01-01 10:05", "Task2"]
        ], columns=["Login", "Start Time", "Task Name"])
        df.to_excel(path, index=False)
        
        with pytest.raises(ValueError, match="Multiple exercises found"):
            parser.parse(str(path), "Sheet1")

    def test_parse_missing_columns_raises(self, parser, tmp_path):
        path = tmp_path / "bad_cols.xlsx"
        df = pd.DataFrame([["val"]], columns=["NotLogin"])
        df.to_excel(path, index=False)
        
        with pytest.raises(ValueError, match="no 'Login' column found"):
            parser.parse(str(path), "Sheet1")

    def test_parse_warnings(self, parser, tmp_path):
        path = tmp_path / "warnings.xlsx"
        # Fewer than LC_MIN_SESSIONS
        df = pd.DataFrame([
            ["user1", "2024-01-01 10:00", "Task1", "Failed"]
        ], columns=["Login", "Start Time", "Task Name", "Status"])
        df.to_excel(path, index=False)
        
        parsed = parser.parse(str(path), "Sheet1")
        assert len(parsed.warnings) >= 2
        assert any("Fewer than" in w for w in parsed.warnings)
        assert any("Failed attempts" in w for w in parsed.warnings)
        assert any("'Total Time (s)' column missing" in w for w in parsed.warnings)
