"""Unit tests for the live VideoFeed widget."""

from unittest.mock import patch

import pytest
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from app.ui.widgets.video_feed import VideoFeed


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    return QApplication.instance() or QApplication([])


def test_hidden_video_feed_still_rebroadcasts_frames(qapp: QApplication) -> None:
    """Secondary preview consumers should receive frames before layout occurs."""
    del qapp
    feed = VideoFeed()
    feed._active = True

    received: list[QImage] = []
    feed.frame_ready.connect(received.append)

    frame = QImage(8, 8, QImage.Format.Format_RGB888)
    frame.fill(0)

    with patch.object(VideoFeed, "size", return_value=QSize(0, 0)):
        feed._on_frame(frame)

    assert received == [frame]
