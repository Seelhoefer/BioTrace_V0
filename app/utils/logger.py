"""Application-wide logging configuration for BioTrace.

All modules obtain their logger via:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)

Logs are written to the console (DEBUG+) and to ``biotrace.log`` (INFO+)
in the working directory.
"""

import logging
import sys
from pathlib import Path

_LOG_FILE = Path("biotrace.log")
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_configured = False


def _configure() -> None:
    """Set up root logger handlers (called once at import time)."""
    global _configured
    if _configured:
        return

    root = logging.getLogger("biotrace")
    root.setLevel(logging.DEBUG)

    # Console handler — INFO and above (DEBUG is too noisy for real-time use).
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_FORMATTER)
    root.addHandler(ch)

    # File handler — INFO and above.
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(_FORMATTER)
    root.addHandler(fh)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'biotrace' namespace.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    _configure()
    # Ensure the name is rooted under 'biotrace' for unified filtering.
    if not name.startswith("biotrace"):
        name = f"biotrace.{name}"
    return logging.getLogger(name)
