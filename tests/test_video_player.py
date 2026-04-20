"""Unit tests for the VideoPlayer widget.

Focuses on load states and placeholder logic.  Since VideoPlayer requires
a GUI event loop and a real MP4 file for many tests, we use a mock-heavy
approach where feasible.
"""

import pytest
from PyQt6.QtWidgets import QApplication
from app.ui.widgets.video_player import VideoPlayer

# Ensure a QApplication exists for widget tests.
@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])

@pytest.fixture()
def player(qapp) -> VideoPlayer:
    return VideoPlayer()

class TestVideoPlayer:
    def test_initial_state_shows_placeholder(self, player: VideoPlayer) -> None:
        """Before loading, a generic prompt should be visible."""
        assert player._display.text() == "Select a session to view recording"
        assert player._controls_container.isHidden()

    def test_load_none_shows_no_recording_placeholder(self, player: VideoPlayer) -> None:
        """Loading None should show the 'no recording' message."""
        player.load(None)
        assert player._display.text() == "No recording available for this session."
        assert player._controls_container.isHidden()

    def test_load_nonexistent_file_shows_no_recording_placeholder(self, player: VideoPlayer) -> None:
        """Loading a missing file should show the 'no recording' message."""
        player.load("/tmp/nonexistent_file_12345.mp4")
        assert player._display.text() == "No recording available for this session."
        assert player._controls_container.isHidden()
