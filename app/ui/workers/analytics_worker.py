"""Background workers for heavy analytics computations.

Prevents the UI thread from blocking during SciPy curve fitting.
"""

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from app.analytics.learning_curve import fit_schmettow, SchmettowFit


class LearningCurveWorker(QObject):
    """Worker for fitting the Schmettow learning curve model.

    Signals:
        finished (SchmettowFit | None): Emitted when fitting is complete.
    """

    finished = pyqtSignal(object)  # carries SchmettowFit | None

    def __init__(self, trials: np.ndarray, errors: np.ndarray, score_max: float) -> None:
        """Initialise the worker with data.

        Args:
            trials: Array of trial numbers.
            errors: Array of error counts.
            score_max: Maximum possible score.
        """
        super().__init__()
        self._trials = trials
        self._errors = errors
        self._score_max = score_max

    @pyqtSlot()
    def run(self) -> None:
        """Perform the fit and emit the result."""
        try:
            fit = fit_schmettow(self._trials, self._errors, self._score_max)
            self.finished.emit(fit)
        except Exception:
            self.finished.emit(None)
