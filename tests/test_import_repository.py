"""Unit tests for ImportRepository.

Verifies persistence and retrieval of external datasets and trials.
"""

import pytest
from app.storage.database import DatabaseManager
from app.storage.import_repository import ImportRepository
from app.analytics.lapsim_parser import ParsedDataset, TrialRecord

@pytest.fixture
def db(tmp_path):
    # Use a real temp file for testing CASCADE and other SQLite features
    db_path = tmp_path / "test_import.db"
    return DatabaseManager(str(db_path))

@pytest.fixture
def repo(db):
    return ImportRepository(db)

@pytest.fixture
def dummy_dataset():
    return ParsedDataset(
        participant="user1",
        exercise="Task1",
        course="Course1",
        source_file="test.xlsx",
        trials=[
            TrialRecord(1, "2024-01-01 10:00", 60.0, 80.0, 2, "Pass"),
            TrialRecord(2, "2024-01-01 10:05", 55.0, 85.0, 1, "Pass"),
        ],
        warnings=[]
    )

class TestImportRepository:
    def test_save_and_get_all(self, repo, dummy_dataset):
        dataset_id = repo.save_dataset(dummy_dataset, "Total Time (s)")
        assert dataset_id > 0
        
        all_ds = repo.get_all_datasets()
        assert len(all_ds) == 1
        assert all_ds[0]["participant"] == "user1"
        assert all_ds[0]["trial_count"] == 2

    def test_get_trials(self, repo, dummy_dataset):
        dataset_id = repo.save_dataset(dummy_dataset, "Total Time (s)")
        trials = repo.get_trials(dataset_id)
        assert len(trials) == 2
        assert trials[0]["trial_number"] == 1
        assert trials[0]["total_time_s"] == 60.0
        assert trials[0]["raw_value"] == 60.0 # because we chose Total Time (s)

    def test_delete_dataset(self, repo, dummy_dataset):
        dataset_id = repo.save_dataset(dummy_dataset, "Total Time (s)")
        assert len(repo.get_all_datasets()) == 1
        
        repo.delete_dataset(dataset_id)
        assert len(repo.get_all_datasets()) == 0
        # Check that trials were deleted via CASCADE
        assert len(repo.get_trials(dataset_id)) == 0
