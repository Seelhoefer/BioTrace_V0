"""BioTrace UI design tokens and global stylesheet.

All colours, fonts, and spacing constants are defined here as module-level
constants.  Import them wherever you need a style value — never hardcode a
hex colour or pixel size inside a widget file.

Usage::

    from app.ui.theme import COLOR_PRIMARY, GLOBAL_STYLESHEET
    self.setStyleSheet(GLOBAL_STYLESHEET)
"""

import qtawesome as qta
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QSize

# ── Colors ────────────────────────────────────────────────────────────
COLOR_PRIMARY         = "#142970"
COLOR_PRIMARY_HOVER   = "#1c3a8c"
COLOR_PRIMARY_SUBTLE  = "#EEF1F9"
COLOR_BACKGROUND      = "#F9FBFF"
COLOR_CARD            = "#FFFFFF"
COLOR_FONT            = "#142970"
COLOR_FONT_MUTED      = "#6B7A9F"
COLOR_FONT_DISABLED   = "#A8B4CE"
COLOR_BORDER          = "#DDE3F0"
COLOR_BORDER_FOCUS    = "#142970"

COLOR_SUCCESS         = "#22C55E"
COLOR_SUCCESS_BG      = "#F0FDF4"
COLOR_WARNING         = "#F59E0B"
COLOR_WARNING_BG      = "#FFFBEB"
COLOR_DANGER          = "#EF4444"
COLOR_DANGER_BG       = "#FEF2F2"

COLOR_CHART_CLI       = "#142970"
COLOR_CHART_RMSSD     = "#22C55E"
COLOR_CHART_PDI       = "#A78BFA"
COLOR_CHART_GRID      = "#E8EDF7"
COLOR_CHART_AXIS      = "#6B7A9F"

# ── Spacing (Rule of 8) ───────────────────────────────────────────────
SPACE_1  =   8
SPACE_2  =  16
SPACE_3  =  24
SPACE_4  =  32
SPACE_5  =  40
SPACE_6  =  48
SPACE_8  =  64
SPACE_12 =  96
SPACE_16 = 128
SPACE_MICRO = 4    # hairline gaps only

# ── Border Radius ─────────────────────────────────────────────────────
RADIUS_SM   =   4
RADIUS_MD   =   8
RADIUS_LG   =  12
RADIUS_XL   =  16
RADIUS_PILL = 999

# ── Font Family ───────────────────────────────────────────────────────
FONT_FAMILY = "Inter"

# ── Font Sizes ────────────────────────────────────────────────────────
FONT_CAPTION      = 11
FONT_SMALL        = 12
FONT_BODY         = 14
FONT_BODY_LARGE   = 16
FONT_SUBTITLE     = 16
FONT_TITLE        = 20
FONT_HEADING_2    = 24
FONT_HEADING_1    = 32
FONT_DISPLAY      = 40
FONT_METRIC_XL    = 48

# ── Font Weights ──────────────────────────────────────────────────────
WEIGHT_REGULAR = 400
WEIGHT_MEDIUM  = 500
WEIGHT_SEMIBOLD = 600
WEIGHT_BOLD    = 700
WEIGHT_EXTRABOLD = 800

# ── Component Dimensions ──────────────────────────────────────────────
SIDEBAR_WIDTH       = 240
SIDEBAR_PADDING_X   =  12
SIDEBAR_PADDING_Y   =  16
SIDEBAR_LOGO_HEIGHT =  64
TOPBAR_HEIGHT       =  64
BTN_HEIGHT_DEFAULT  =  40
INPUT_HEIGHT        =  40
NAV_ITEM_HEIGHT     =  48
CALIBRATION_CTA_WIDTH = 160
CALIBRATION_CTA_HEIGHT = 48
ICON_SIZE_INLINE    =  16
ICON_SIZE_DEFAULT   =  20
ICON_SIZE_NAV       =  20
ICON_SIZE_NAV_LOGO  =  32
CARD_PADDING        =  24
CARD_RADIUS         =  12
METRIC_CARD_WIDTH   = 280
METRIC_CARD_HEIGHT  = 168
METRIC_CARD_COMPACT_HEIGHT = 96

# ── Layout ────────────────────────────────────────────────────────────
CONTENT_PADDING_H   =  32
CONTENT_PADDING_V   =  24
GRID_GUTTER         =  24
CHART_HEIGHT_FULL   = 240
CHART_HEIGHT_HALF   = 200
CHART_HEIGHT_TIMELINE = 320
CALIBRATION_CARD_WIDTH = 640

# ── Shadows (as QSS string) ───────────────────────────────────────────
SHADOW_COLOR_SM = (20, 41, 112, 15)
SHADOW_COLOR_MD = (20, 41, 112, 20)
SHADOW_COLOR_LG = (20, 41, 112, 26)
SHADOW_BLUR_SM  =  6
SHADOW_BLUR_MD  = 16
SHADOW_BLUR_LG  = 32
SHADOW_OFFSET_SM = (0, 1)
SHADOW_OFFSET_MD = (0, 4)
SHADOW_OFFSET_LG = (0, 8)

GLOBAL_STYLESHEET: str = f"""
/* ── Root window ─────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {COLOR_BACKGROUND};
    color: {COLOR_FONT};
    font-family: "{FONT_FAMILY}", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: {FONT_BODY}px;
}}

/* ── Cards / panels ──────────────────────────────────────────────── */
QFrame#card, QWidget#card {{
    background-color: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: {RADIUS_LG}px;
}}

/* ── Primary button ──────────────────────────────────────────────── */
QPushButton {{
    background-color: {COLOR_PRIMARY};
    color: #FFFFFF;
    border: none;
    border-radius: {RADIUS_MD}px;
    padding: 10px 16px;
    font-size: {FONT_BODY}px;
    font-weight: {WEIGHT_MEDIUM};
}}
QPushButton:hover {{
    background-color: {COLOR_PRIMARY_HOVER};
}}
QPushButton:pressed {{
    background-color: {COLOR_PRIMARY_HOVER};
    padding-top: 11px;
    padding-bottom: 9px;
}}
QPushButton:disabled {{
    background-color: {COLOR_FONT_DISABLED};
    color: #FFFFFF;
}}

/* ── Secondary / outline button ─────────────────────────────────── */
QPushButton#secondary {{
    background-color: {COLOR_CARD};
    color: {COLOR_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: {RADIUS_MD}px;
    padding: 10px 16px;
}}
QPushButton#secondary:hover {{
    background-color: {COLOR_PRIMARY_SUBTLE};
}}

/* ── Labels ──────────────────────────────────────────────────────── */
QLabel {{
    color: {COLOR_FONT};
    background-color: transparent;
}}
QLabel#heading {{
    font-size: {FONT_HEADING_2}px;
    font-weight: {WEIGHT_BOLD};
    color: {COLOR_FONT};
}}
QLabel#subheading {{
    font-size: {FONT_SUBTITLE}px;
    font-weight: {WEIGHT_SEMIBOLD};
    color: {COLOR_FONT};
}}
QLabel#muted {{
    color: {COLOR_FONT_MUTED};
    font-size: {FONT_SMALL}px;
}}
QLabel#sidebar_logo_label {{
    color: {COLOR_FONT};
    font-size: {FONT_CAPTION}px;
    font-weight: {WEIGHT_SEMIBOLD};
}}
QLabel#sidebar_section_label {{
    color: {COLOR_FONT_MUTED};
    font-size: {FONT_CAPTION}px;
    font-weight: {WEIGHT_SEMIBOLD};
    letter-spacing: 1px;
}}
QLabel#metric_value {{
    font-size: {FONT_METRIC_XL}px;
    font-weight: {WEIGHT_EXTRABOLD};
    color: {COLOR_PRIMARY};
}}

/* ── Table ───────────────────────────────────────────────────────── */
QTableWidget {{
    background-color: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: {RADIUS_LG}px;
    gridline-color: {COLOR_BORDER};
}}
QTableWidget::item:selected {{
    background-color: {COLOR_PRIMARY_SUBTLE};
    color: {COLOR_PRIMARY};
}}
QHeaderView::section {{
    background-color: {COLOR_CARD};
    color: {COLOR_FONT_MUTED};
    font-size: {FONT_BODY}px;
    font-weight: {WEIGHT_MEDIUM};
    border: none;
    border-bottom: 1px solid {COLOR_BORDER};
    padding: 6px;
    height: {SPACE_5}px;
}}
QTableView::item {{
    height: {SPACE_6}px;
}}

/* ── Scroll bars ─────────────────────────────────────────────────── */
QScrollBar:vertical {{
    width: 8px;
    background: transparent;
}}
QScrollBar::handle:vertical {{
    background: {COLOR_BORDER};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* ── Line edits / spin boxes / combos ───────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: {RADIUS_MD}px;
    padding: 0px 12px;
    padding-right: 32px;
    height: {INPUT_HEIGHT}px;
    color: {COLOR_FONT};
    font-size: {FONT_BODY}px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {COLOR_BORDER_FOCUS};
}}
QComboBox::drop-down {{
    border: none;
    width: 32px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid rgba(0,0,0,0);
    border-right: 5px solid rgba(0,0,0,0);
    border-top: 5px solid {COLOR_FONT_MUTED};
    width: 0px;
    height: 0px;
    subcontrol-origin: content;
    subcontrol-position: center;
}}
QComboBox::down-arrow:on {{
    border-top: none;
    border-bottom: 5px solid {COLOR_FONT_MUTED};
}}
QComboBox QAbstractItemView {{
    background-color: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    selection-background-color: {COLOR_PRIMARY_SUBTLE};
    selection-color: {COLOR_PRIMARY};
    outline: none;
    padding: 4px;
}}

/* ── Sliders (NASA-TLX) ──────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 6px;
    background: {COLOR_BORDER};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {COLOR_PRIMARY};
    border: 2px solid #FFFFFF;
    width: 20px;
    height: 20px;
    margin: -7px 0;
    border-radius: 10px;
}}
QSlider::sub-page:horizontal {{
    background: {COLOR_PRIMARY};
    border-radius: 3px;
}}

/* ── Sidebar ─────────────────────────────────────────────────────── */
QWidget#sidebar {{
    background-color: {COLOR_BACKGROUND};
    border-right: 1px solid {COLOR_BORDER};
}}
QPushButton#nav_button {{
    background-color: transparent;
    color: {COLOR_FONT_MUTED};
    border: none;
    border-radius: {RADIUS_MD}px;
    padding: 0px 12px;
    height: {NAV_ITEM_HEIGHT}px;
    text-align: left;
    font-size: {FONT_BODY}px;
    font-weight: {WEIGHT_MEDIUM};
}}
QPushButton#nav_button:hover {{
    background-color: {COLOR_PRIMARY_SUBTLE};
    color: {COLOR_PRIMARY};
}}
QPushButton#nav_button:checked {{
    background-color: {COLOR_PRIMARY_SUBTLE};
    color: {COLOR_PRIMARY};
    border: none;
}}

/* ── Separators ──────────────────────────────────────────────────── */
QFrame[frameShape="4"],
QFrame[frameShape="5"] {{
    color: {COLOR_BORDER};
}}
"""


def get_icon(name: str, color: str = COLOR_PRIMARY, size: int = 20) -> QIcon:
    """Create a QIcon from the Phosphor library (ph.*) or others via qtawesome.

    Args:
        name: Icon name (e.g., 'ph.chart-bar-fill').
        color: Hex color string.
        size: Icon size in pixels.

    Returns:
        A QIcon object.
    """
    try:
        return qta.icon(name, color=color)
    except Exception:
        # Fallback to a basic circle or empty icon if the requested one is missing
        try:
            return qta.icon("ph.circle-fill", color=color)
        except Exception:
            return QIcon()