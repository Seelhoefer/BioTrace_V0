"""FlowchartCard — animated scientific pipeline flowchart widget for BioTrace.

Each card displays a horizontal row of icon nodes connected by animated
dotted lines (flowing-dots effect). Clicking a node expands an inline
detail panel showing the step's formula, description, and citation.

Architecture:
    FlowchartCard (QFrame#card)
      ├── _FlowchartCanvas   ← QPainter + QTimer, emits node_clicked(int)
      └── _DetailPanel       ← hidden QFrame, shown on node click
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.theme import (
    CARD_PADDING,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_FONT,
    COLOR_FONT_MUTED,
    COLOR_PRIMARY,
    COLOR_PRIMARY_SUBTLE,
    COLOR_WARNING,
    COLOR_WARNING_BG,
    FONT_BODY,
    FONT_CAPTION,
    FONT_FAMILY,
    FONT_SMALL,
    ICON_SIZE_INLINE,
    RADIUS_MD,
    SPACE_1,
    SPACE_2,
    get_icon,
)

# ── Constants ─────────────────────────────────────────────────────────────
_NODE_SIZE   = 56   # px — icon background square
_ICON_SIZE   = 28   # px — icon inside the square
_LABEL_GAP   = 12   # px — gap between node bottom and label
_LABEL_HEIGHT = 20  # px
_NODE_CENTER_Y_RATIO = 0.5    # Center vertically
_DASH_PERIOD = 12.0           # wraps dash offset at this value


# ── NodeDef ───────────────────────────────────────────────────────────────

@dataclass
class NodeDef:
    """Definition for a single flowchart step.

    Attributes:
        icon: qtawesome icon name (e.g. ``"ph.heart-fill"``).
        label: Short label shown below the icon (≤12 chars works best).
        formula: Unicode math formula string shown in the detail panel.
        description: 1–2 sentence plain-English explanation.
        reference: Full citation string, or empty string if none.
        is_threshold: If True, renders with warning colours (threshold gate).
    """
    icon: str
    label: str
    formula: str
    description: str
    reference: str
    is_threshold: bool = False


# ── _FlowchartCanvas ──────────────────────────────────────────────────────

class _FlowchartCanvas(QWidget):
    """Custom-painted canvas showing nodes connected by animated dotted lines.

    Signals:
        node_clicked (int): Emitted with the 0-based index of the clicked node.
    """

    node_clicked = pyqtSignal(int)

    def __init__(self, nodes: list[NodeDef], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._nodes = nodes
        self._dash_offset: float = 0.0
        self._pixmaps: list[QPixmap] = []

        self.setMinimumHeight(120) # Slightly taller for centered layout
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._timer = QTimer(self)
        self._timer.setInterval(25)   # ~40 fps
        self._timer.timeout.connect(self._tick)

        self._prerender_icons()

    # ------------------------------------------------------------------
    # Animation control
    # ------------------------------------------------------------------

    def start_animation(self) -> None:
        """Start the flowing-dots animation."""
        self._timer.start()

    def stop_animation(self) -> None:
        """Pause the animation (saves CPU when the page is hidden)."""
        self._timer.stop()

    def _tick(self) -> None:
        """Advance the dash offset and request a repaint."""
        # Decrementing makes dots move left-to-right (forward)
        # 0.2 is much slower than 1.0
        self._dash_offset = (self._dash_offset - 0.2) % _DASH_PERIOD
        self.update()

    # ------------------------------------------------------------------
    # Icon pre-rendering
    # ------------------------------------------------------------------

    def _prerender_icons(self) -> None:
        """Render all node icons to QPixmap once at construction time.

        Falls back to ``ph.circle-fill`` if the requested icon name is not
        found in qtawesome, so the app never crashes on a missing icon.
        """
        self._pixmaps.clear()
        for nd in self._nodes:
            color = COLOR_WARNING if nd.is_threshold else COLOR_PRIMARY
            try:
                pm = get_icon(nd.icon, color=color, size=_ICON_SIZE).pixmap(
                    QSize(_ICON_SIZE, _ICON_SIZE)
                )
            except Exception:
                try:
                    pm = get_icon("ph.circle-fill", color=color, size=_ICON_SIZE).pixmap(
                        QSize(_ICON_SIZE, _ICON_SIZE)
                    )
                except Exception:
                    pm = QPixmap(_ICON_SIZE, _ICON_SIZE)
                    pm.fill(QColor(color))
            self._pixmaps.append(pm)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _node_centers(self) -> list[tuple[int, int]]:
        """Return (cx, cy) for each node in the current widget size."""
        n = len(self._nodes)
        if n == 0:
            return []
        w = self.width()
        cy = int(self.height() * _NODE_CENTER_Y_RATIO)
        gap = w / n
        return [(int(gap * i + gap / 2), cy) for i in range(n)]

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        centers = self._node_centers()
        n = len(centers)
        half = _NODE_SIZE // 2

        # 1. Connector lines (animated dashes + arrowhead)
        dash_pen = QPen(QColor(COLOR_PRIMARY))
        dash_pen.setWidth(2)
        dash_pen.setStyle(Qt.PenStyle.CustomDashLine)
        dash_pen.setDashPattern([6.0, 6.0])
        dash_pen.setDashOffset(self._dash_offset)

        arrow_pen = QPen(QColor(COLOR_PRIMARY))
        arrow_pen.setWidth(2)

        for i in range(n - 1):
            cx1, cy1 = centers[i]
            cx2, cy2 = centers[i + 1]
            x1 = cx1 + half + 6
            x2 = cx2 - half - 6
            if x2 > x1:
                # Shadow/background line for better visibility
                bg_pen = QPen(QColor(COLOR_PRIMARY_SUBTLE))
                bg_pen.setWidth(2)
                painter.setPen(bg_pen)
                painter.drawLine(x1, cy1, x2, cy1)

                painter.setPen(dash_pen)
                painter.drawLine(x1, cy1, x2, cy1)

                # Arrowhead at destination end
                painter.setPen(arrow_pen)
                a = 6
                painter.drawLine(x2, cy1, x2 - a, cy1 - a)
                painter.drawLine(x2, cy1, x2 - a, cy1 + a)

        # 2. Node icons (centered, no background)
        for i, (cx, cy) in enumerate(centers):
            pm = self._pixmaps[i]
            painter.drawPixmap(cx - pm.width() // 2, cy - pm.height() // 2, pm)

        # 3. Labels below nodes
        label_font = QFont(FONT_FAMILY)
        label_font.setPixelSize(FONT_CAPTION)
        label_font.setWeight(600)
        painter.setFont(label_font)
        painter.setPen(QPen(QColor(COLOR_FONT)))

        for i, (cx, cy) in enumerate(centers):
            label_y = cy + half + _LABEL_GAP
            painter.drawText(
                QRect(cx - 60, label_y, 120, _LABEL_HEIGHT),
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
                self._nodes[i].label,
            )

        painter.end()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Emit node_clicked(i) when the user clicks inside a node rect."""
        pos = event.position()
        click_x, click_y = pos.x(), pos.y()
        half = _NODE_SIZE // 2

        for i, (cx, cy) in enumerate(self._node_centers()):
            if abs(click_x - cx) <= half and abs(click_y - cy) <= half:
                self.node_clicked.emit(i)
                return
        super().mousePressEvent(event)


# ── _DetailPanel ──────────────────────────────────────────────────────────

class _DetailPanel(QFrame):
    """Inline panel that appears below the canvas when a node is clicked.

    Shows the step name, formula (monospace block), plain-English
    description, and scientific reference.

    Signals:
        close_requested: Emitted when the user clicks the ✕ button.
    """

    close_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background-color: {COLOR_PRIMARY_SUBTLE}; "
            f"border: 1px solid {COLOR_BORDER}; border-radius: {RADIUS_MD}px; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_2, SPACE_2, SPACE_2, SPACE_2)
        layout.setSpacing(SPACE_1)

        # ── Top row: step name + close button ────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(SPACE_1)

        self._step_lbl = QLabel()
        self._step_lbl.setStyleSheet(
            f"font-size: {FONT_BODY}px; font-weight: 700; color: {COLOR_FONT}; "
            f"background: transparent; border: none;"
        )
        top_row.addWidget(self._step_lbl)
        top_row.addStretch()

        close_btn = QPushButton()
        try:
            close_btn.setIcon(get_icon("ph.x-fill", color=COLOR_FONT_MUTED))
        except Exception:
            close_btn.setText("✕")
        close_btn.setIconSize(QSize(ICON_SIZE_INLINE, ICON_SIZE_INLINE))
        close_btn.setObjectName("secondary")
        close_btn.setFixedSize(28, 28)
        close_btn.setStyleSheet(
            f"QPushButton#secondary {{ background: transparent; border: none; color: {COLOR_FONT_MUTED}; }}"
            f"QPushButton#secondary:hover {{ background: {COLOR_BORDER}; }}"
        )
        close_btn.clicked.connect(self.close_requested)
        top_row.addWidget(close_btn)
        layout.addLayout(top_row)

        # ── Formula block (monospace) ────────────────────────────────
        formula_frame = QFrame()
        formula_frame.setStyleSheet(
            f"QFrame {{ background: {COLOR_CARD}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: {RADIUS_MD}px; }}"
        )
        ff_layout = QVBoxLayout(formula_frame)
        ff_layout.setContentsMargins(SPACE_2, SPACE_1, SPACE_2, SPACE_1)

        self._formula_lbl = QLabel()
        self._formula_lbl.setWordWrap(True)
        self._formula_lbl.setStyleSheet(
            f"font-family: 'Courier New', monospace; font-size: {FONT_BODY}px; "
            f"color: {COLOR_PRIMARY}; background: transparent; border: none;"
        )
        ff_layout.addWidget(self._formula_lbl)
        layout.addWidget(formula_frame)

        # ── Description ──────────────────────────────────────────────
        self._desc_lbl = QLabel()
        self._desc_lbl.setWordWrap(True)
        self._desc_lbl.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {FONT_BODY}px; "
            f"background: transparent; border: none;"
        )
        layout.addWidget(self._desc_lbl)

        # ── Reference ────────────────────────────────────────────────
        self._ref_lbl = QLabel()
        self._ref_lbl.setWordWrap(True)
        self._ref_lbl.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_SMALL}px; "
            f"font-style: italic; background: transparent; border: none;"
        )
        layout.addWidget(self._ref_lbl)

        self.setVisible(False)

    def show_node(self, node: NodeDef) -> None:
        """Populate with node data and make the panel visible.

        Args:
            node: The :class:`NodeDef` whose details to display.
        """
        self._step_lbl.setText(node.label)
        self._formula_lbl.setText(node.formula)
        self._desc_lbl.setText(node.description)
        if node.reference:
            self._ref_lbl.setText(f"Reference: {node.reference}")
            self._ref_lbl.setVisible(True)
        else:
            self._ref_lbl.setVisible(False)
        self.setVisible(True)

    def hide_panel(self) -> None:
        """Hide the detail panel."""
        self.setVisible(False)


# ── FlowchartCard ─────────────────────────────────────────────────────────

class FlowchartCard(QFrame):
    """A titled card containing an animated flowchart and a click-to-reveal
    formula panel.

    Args:
        title: Card heading (e.g. ``"Stress"``).
        subtitle: Muted subtitle (e.g. ``"HRV · RMSSD · Δ from Baseline"``).
        nodes: Ordered list of :class:`NodeDef` objects.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        title: str,
        subtitle: str,
        nodes: list[NodeDef],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet(
            f"QFrame#card {{ background-color: transparent; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: {RADIUS_MD}px; }}"
        )
        self._nodes = nodes
        self._active_index = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(CARD_PADDING, CARD_PADDING, CARD_PADDING, CARD_PADDING)
        layout.setSpacing(SPACE_1)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("subheading")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_lbl)

        sub_lbl = QLabel(subtitle)
        sub_lbl.setObjectName("muted")
        sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub_lbl)

        layout.addSpacing(SPACE_2)

        self._canvas = _FlowchartCanvas(nodes)
        layout.addWidget(self._canvas)

        self._detail_panel = _DetailPanel()
        self._detail_panel.close_requested.connect(self._close_detail)
        layout.addWidget(self._detail_panel)

        self._canvas.node_clicked.connect(self._on_node_clicked)

    # ------------------------------------------------------------------
    # Animation control (delegated to canvas)
    # ------------------------------------------------------------------

    def start_animation(self) -> None:
        """Start the flowing-dots animation on the canvas."""
        self._canvas.start_animation()

    def stop_animation(self) -> None:
        """Stop the flowing-dots animation."""
        self._canvas.stop_animation()

    # ------------------------------------------------------------------
    # Node click handling
    # ------------------------------------------------------------------

    def _on_node_clicked(self, index: int) -> None:
        """Show the detail panel for the clicked node, or toggle it off."""
        if self._active_index == index:
            self._close_detail()
        else:
            self._active_index = index
            self._detail_panel.show_node(self._nodes[index])

    def _close_detail(self) -> None:
        """Hide the detail panel and reset active node tracking."""
        self._active_index = -1
        self._detail_panel.hide_panel()
