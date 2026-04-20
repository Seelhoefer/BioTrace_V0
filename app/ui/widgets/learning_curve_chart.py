"""Learning curve chart widget using pyqtgraph.

Displays a trainee's performance trajectory (dots), a fitted Schmettow curve
(solid line), and a future projection (dashed line).
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QStackedWidget
from PyQt6.QtCore import Qt

from app.analytics.learning_curve import SchmettowFit, SessionDataPoint, predict_at_trial, mastery_percent
from app.utils.config import SCORE_MAX, LC_MIN_SESSIONS
from app.ui.theme import (
    COLOR_PRIMARY,
    COLOR_CARD,
    COLOR_BORDER,
    COLOR_FONT,
    COLOR_FONT_MUTED,
    COLOR_DANGER,
    FONT_CAPTION,
    FONT_BODY,
    SPACE_1,
    RADIUS_LG,
)


class LearningCurveChart(QWidget):
    """Chart widget for visualizing the Schmettow learning curve.

    Args:
        show_position_marker: If True, draws a vertical line at the last trial.
        parent: Optional parent widget.
    """

    def __init__(self, show_position_marker: bool = False, transparent: bool = False, parent: QWidget | None = None) -> None:
        """Initialise the learning curve chart."""
        super().__init__(parent)
        self._show_position_marker = show_position_marker
        self._transparent = transparent
        self._init_ui()

    def _init_ui(self) -> None:
        """Build the chart layout with a plot/placeholder stack."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_1)

        # Stack for switching between the plot and a placeholder label
        self._stack = QStackedWidget()

        # Plot container
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMouseEnabled(x=False, y=False)
        self._plot_widget.hideButtons()
        if self._transparent:
            self._plot_widget.setBackground((0, 0, 0, 0))
        else:
            self._plot_widget.setBackground(COLOR_CARD)
        self._plot_widget.showGrid(x=False, y=True, alpha=0.3)

        # Style axes to match theme
        for axis_name in ("bottom", "left"):
            axis = self._plot_widget.getAxis(axis_name)
            axis.setPen(pg.mkPen(color=COLOR_BORDER, width=1))
            axis.setTextPen(pg.mkPen(color=COLOR_FONT_MUTED))

        self._plot_widget.getAxis("bottom").setLabel("Session Number", color=COLOR_FONT_MUTED)
        self._plot_widget.getAxis("left").setLabel("Performance Score", color=COLOR_FONT_MUTED)

        self._stack.addWidget(self._plot_widget)

        # Placeholder for insufficient data
        self._placeholder = QLabel("Not enough sessions to model your learning curve.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_BODY}px; "
            f"background-color: {COLOR_CARD}; border: 1px dashed {COLOR_BORDER}; "
            f"border-radius: {RADIUS_LG}px;"
        )
        self._stack.addWidget(self._placeholder)

        layout.addWidget(self._stack)
        self._stack.setCurrentWidget(self._placeholder)

    def update_data(
        self,
        series: list[SessionDataPoint],
        fit: SchmettowFit | None,
        metric_label: str = "Performance Score",
        y_axis_inverted: bool = False,
    ) -> None:
        """Update the chart with new session data and an optional model fit.

        Args:
            series: List of SessionDataPoint objects (the training history).
            fit: A SchmettowFit object if the model was successfully fitted,
                 else None.
            metric_label: Label for the y-axis (e.g., "Total Time (s)").
            y_axis_inverted: If True, lower values are better (downward curve).
        """
        self._plot_widget.clear()
        self._plot_widget.getAxis("left").setLabel(metric_label, color=COLOR_FONT_MUTED)

        if not series:
            self._stack.setCurrentWidget(self._placeholder)
            self._placeholder.setText("No session data available.")
            return

        trials = np.array([dp.trial for dp in series])
        
        # Extract the raw backend values. Note that lapsim_metrics.py calculates 
        # a composite error score (0-100, where 100 is worst) and stores it here.
        raw_backend_values = np.array([dp.performance_score for dp in series])

        # Align the raw data points with the desired display domain
        if y_axis_inverted:
            # Domain: Lower is better (e.g., Time, Error Score)
            display_values = raw_backend_values
        else:
            # Domain: Higher is better (Performance Score)
            # Invert the points so that 100 represents the best possible performance
            display_values = SCORE_MAX - raw_backend_values

        if fit is None:
            self._stack.setCurrentWidget(self._placeholder)
            needed = LC_MIN_SESSIONS - len(series)
            if needed > 0:
                self._placeholder.setText(
                    f"Complete {needed} more session{'s' if needed > 1 else ''} "
                    "to unlock analysis."
                )
            else:
                self._placeholder.setText("Learning curve could not be fitted.")
            return

        self._stack.setCurrentWidget(self._plot_widget)

        # 1. Actual performance (scatter dots)
        self._plot_widget.plot(
            trials, display_values, pen=None, symbol="o", symbolSize=10,
            symbolBrush=pg.mkBrush(COLOR_PRIMARY), symbolPen=pg.mkPen(COLOR_PRIMARY)
        )

        # 2. Fitted curve (solid line)
        t_fit = np.linspace(1, float(trials[-1]), 100)
        
        # Project the curve based on whether we are in the error domain or score domain.
        def get_curve_val(t):
            err = fit.scale * (1 - fit.leff)**(t + fit.pexp) + fit.maxp
            if y_axis_inverted:
                # Time domain or Error Score domain (lower is better, 0 is target)
                return err
            else:
                # Score domain. Use the SCORE_MAX constant directly.
                return SCORE_MAX - err

        p_fit = [get_curve_val(t) for t in t_fit]
        self._plot_widget.plot(t_fit, p_fit, pen=pg.mkPen(color=COLOR_PRIMARY, width=2.5))

        # 3. 3-trial projection (dashed line)
        t_proj = np.linspace(float(trials[-1]), float(trials[-1] + 3), 50)
        p_proj = [get_curve_val(t) for t in t_proj]
        self._plot_widget.plot(
            t_proj, p_proj,
            pen=pg.mkPen(color=COLOR_FONT_MUTED, width=2, style=Qt.PenStyle.DashLine)
        )

        # 4. Asymptotic ceiling/floor (horizontal dashed line)
        asymptote = fit.maxp if y_axis_inverted else fit.maxp_performance
        asymptote_line = pg.InfiniteLine(
            pos=asymptote, angle=0,
            pen=pg.mkPen(color=COLOR_DANGER, width=1, style=Qt.PenStyle.DashLine)
        )
        self._plot_widget.addItem(asymptote_line)

        # Label for the asymptote line
        label_text = "Potential (floor)" if y_axis_inverted else "Your potential"
        asymptote_label = pg.TextItem(label_text, color=COLOR_DANGER, anchor=(1, 1))
        asymptote_label.setPos(float(trials[-1] + 3), asymptote)
        self._plot_widget.addItem(asymptote_label)

        # 5. Position marker (optional vertical line at last trial)
        if self._show_position_marker:
            marker = pg.InfiniteLine(
                pos=float(trials[-1]), angle=90,
                pen=pg.mkPen(color=COLOR_PRIMARY, width=1)
            )
            self._plot_widget.addItem(marker)

        # Adjust range to include the projection and asymptote
        self._plot_widget.setXRange(0.5, float(trials[-1] + 3.5), padding=0)
        
        # Combine all relevant Y values to find the viewable range
        all_y = np.concatenate([display_values, p_fit, p_proj, [asymptote]])
        v_min = all_y.min() * 0.95
        v_max = all_y.max() * 1.05
        
        # Ensure we show a reasonable range even if data is very tight
        if v_max - v_min < 1.0:
            v_min -= 5.0
            v_max += 5.0
            
        self._plot_widget.setYRange(v_min, v_max, padding=0)