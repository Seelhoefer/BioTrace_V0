# BioTrace

BioTrace is a desktop application for real time physiological biofeedback during laparoscopic surgical training. It was built as part of a university research project at the TSS Lab.

The idea is simple: while a trainee practices on a laparoscopic simulator, BioTrace captures their heart rate variability and pupil dilation in real time, records the endoscope video, and tracks wall contacts as errors. After each session the trainee can review everything together on a synchronized timeline, and over multiple sessions the app fits a learning curve so both trainee and mentor can see how performance is developing.


## What it does

BioTrace runs a calibration phase first to establish a physiological baseline, then switches into a live session view. During the session it shows real time stress and cognitive workload indicators derived from ECG based HRV (RMSSD) and pupil dilation. A composite Cognitive Load Index combines both signals.

The live view has two modes. The biofeedback dashboard shows detailed charts and gauges. The camera HUD mode shows the endoscope feed fullscreen with a minimal overlay so it stays out of the way during actual training.

After a session ends, the post session view lets you replay the recorded video side by side with a timeline of all biometric data. Clicking on a spike in the chart jumps the video to that exact moment. The view also shows summary statistics like total time, error count, and stress events.

On the main dashboard, BioTrace aggregates data across all sessions. It fits a Schmettow parametric learning curve to the trainee's performance data, combining speed and accuracy into a single metric using a penalty time approach (each wall contact adds a time penalty). The curve has interpretable parameters for previous experience, learning efficiency, and maximum achievable performance, which makes it useful for research and mentoring conversations.

There is also an Excel import view for bringing in data from external sources like LapSim exports, so the learning curve can include sessions recorded outside of BioTrace.


## Hardware

BioTrace is designed to work with a Raspberry Pi Pico running YLab Zero firmware for ECG acquisition and a USB camera for pupil tracking. The endoscope camera captures the training video.

If no hardware is connected, the app falls back to mock sensors automatically, so you can develop and test the UI without any physical devices.

Hardware settings like serial ports and baud rates are configured in `app/utils/config.py`.


## Setup

You need Python 3.11 or newer.

Clone the repository, create a virtual environment, and install the dependencies:

```
git clone https://github.com/fly-withme/biotrace.git
cd biotrace
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\activate` instead.

Then start the app:

```
python main.py
```

The entry point auto detects the virtual environment, so `python main.py` also works without activating it manually.


## Project layout

```
biotrace/
    main.py                 Application entry point
    app/
        analytics/          Learning curve fitting, LapSim import parsing
        core/               Session lifecycle, metric algorithms
        hardware/           Device drivers (Pico ECG, eye tracker, error counter)
        processing/         Real time signal processing (HRV, pupil, stress, LHIPA)
        storage/            SQLite database and Excel export
        ui/
            theme.py        Design tokens (colors, fonts, spacing)
            views/          Main pages (dashboard, live, calibration, post session, settings, import)
            widgets/        Reusable components (charts, gauges, video player, needle gauge)
    tests/                  Pytest suite
```


## Configuration

All thresholds, hardware ports, algorithm weights, and timing constants live in `app/utils/config.py`. If you need to change behavior, that is the only file you should have to touch.


## Tests

```
python -m pytest tests/
```


## License

2026 TSS Lab. Built for surgical education research.
