"""
run_dev.py — Development runner with auto-reload

Starts main.py and restarts the process automatically on code changes.
Usage:
    python3 run_dev.py
"""
import os
import subprocess
import sys

# --- Virtual Environment Auto-Activation ---
def _ensure_venv() -> None:
    """If not running in a venv, try to re-run with the local .venv python."""
    # Check if we are already in a virtual environment
    in_venv = sys.prefix != sys.base_prefix or hasattr(sys, "real_prefix")
    if in_venv:
        return

    # Look for .venv in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, ".venv")

    if os.path.isdir(venv_dir):
        # Determine python executable path (Windows vs macOS/Linux)
        if os.name == "nt":
            python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            python_exe = os.path.join(venv_dir, "bin", "python")

        if os.path.isfile(python_exe):
            # Re-execute the current script using the venv python
            os.execv(python_exe, [python_exe] + sys.argv)

_ensure_venv()
# -------------------------------------------

import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCHED_EXTENSIONS = {'.py'}
WATCHED_DIRS = ['app', 'main.py']

class RestartHandler(FileSystemEventHandler):
    def __init__(self, restart_callback):
        super().__init__()
        self.restart_callback = restart_callback

    def on_any_event(self, event):
        if event.is_directory:
            return
        if any(event.src_path.endswith(ext) for ext in WATCHED_EXTENSIONS):
            self.restart_callback()

def run_main_py():
    return subprocess.Popen([sys.executable, 'main.py'])

def main():
    process = run_main_py()
    needs_restart = False

    def restart():
        nonlocal needs_restart
        needs_restart = True
        process.terminate()

    event_handler = RestartHandler(restart)
    observer = Observer()
    for path in WATCHED_DIRS:
        observer.schedule(event_handler, str(Path(path).resolve()), recursive=True)
    observer.start()

    try:
        while True:
            if needs_restart:
                process.wait()
                process = run_main_py()
                needs_restart = False
            time.sleep(0.5)
    except KeyboardInterrupt:
        process.terminate()
    finally:
        observer.stop()
        observer.join()

if __name__ == '__main__':
    main()
