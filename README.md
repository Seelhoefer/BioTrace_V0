# BioTrace

BioTrace is a desktop application designed for real-time physiological biofeedback and performance analysis during laparoscopic surgical training. It provides trainees and researchers with immediate insights into stress levels and cognitive workload, synchronized with live video recordings of the training session.

## 🚀 Key Features

*   **Real-time Biofeedback**: Monitors HRV (RMSSD), Pupil Dilation Index (PDI), and a composite Cognitive Load Index (CLI).
*   **Dual-Mode Live View**:
    *   **Biofeedback Dashboard**: High-fidelity real-time charts and gauges focusing on physiological metrics.
    *   **Camera HUD**: Full-screen video feed with a minimalist, non-intrusive heads-up display of core metrics.
*   **Individual Session Analysis**: Redesigned post-session dashboard featuring:
    *   **Synchronized Playback**: Video player linked to a biometric timeline chart—click any data spike to jump to that moment in the video.
    *   **KPI Summary**: Quick-glance metrics for Performance, Total Time, and Error Rate using intuitive donut gauges.
    *   **Session Management**: Rename sessions directly from the dashboard for better organization.
*   **Data Export**: Comprehensive export of session data to Excel for longitudinal research and statistical analysis.
*   **Hardware Integration**: Supports Raspberry Pi Pico (ECG) and Pupil Labs eye trackers, with built-in mock sensors for development without hardware.

## 🛠 Installation

### Prerequisites
- Python 3.11 or newer
- [Optional] USB ECG Sensor (Pico-based) or Pupil Labs Eye Tracker

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/fly-withme/biotrace.git
   cd biotrace
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Getting Started

### Running the Application
Launch the main interface:
```bash
python main.py
```

### Development Mode
The application automatically falls back to **Mock Sensors** if hardware is not detected, allowing for UI and logic development without physical devices.

## 📂 Project Structure

```text
biotrace/
├── main.py                # Application entry point
├── app/
│   ├── analytics/         # Learning curve fitting and LapSim parsers
│   ├── core/              # Session lifecycle and metric algorithms (CLI, RMSSD)
│   ├── hardware/          # Device drivers (Pico ECG, Eye Tracker, Mock)
│   ├── processing/        # Real-time signal processing pipelines
│   ├── storage/           # SQLite database management and data export
│   └── ui/                # PyQt6 Views and custom Widgets
│       ├── theme.py       # Design tokens (Colors, Fonts, Spacing)
│       └── views/         # Major application pages (Dashboard, Live, Post-Session)
└── tests/                 # Comprehensive PyTest suite
```

## 🎨 Design System
BioTrace follows a strict design system defined in `app/ui/theme.py`. 
- **Rule of 8**: All spacing and dimensions are multiples of 8px.
- **Color Palette**: Deep navy backgrounds with vibrant primary blues and status-aware accents (Success Green, Warning Orange, Danger Red).
- **Typography**: Focused on high-legibility sans-serif fonts for data-heavy displays.

## 📝 License
© 2026 TSS Lab. Designed for surgical education research. All rights reserved.
