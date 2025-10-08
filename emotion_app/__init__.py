# emotion_app/__init__.py
"""
Public package interface for emotion_app.

Exports a stable API for the rest of the app, regardless of internal function
names in detector.py (detect_emotions vs emotion_detector).
"""

from __future__ import annotations
import importlib

# --- Load detector lazily to avoid hard import failures at package import time ---
_det = importlib.import_module(".detector", __name__)

# Export dataclass (required)
try:
    EmotionResult = getattr(_det, "EmotionResult")
except AttributeError as exc:
    raise ImportError("EmotionResult not found in emotion_app.detector") from exc

# Find a callable detector by common names and expose it as `emotion_detector`
_emotion_fn = (
    getattr(_det, "emotion_detector", None)
    or getattr(_det, "detect_emotions", None)
    or getattr(_det, "analyze_emotions", None)
)
if _emotion_fn is None or not callable(_emotion_fn):
    raise ImportError(
        "No callable emotion detector found in emotion_app.detector. "
        "Define def detect_emotions(text: str) -> EmotionResult (or emotion_detector)."
    )

def emotion_detector(text: str):
    """Stable public entry point."""
    return _emotion_fn(text)

# --- Formatter (optional but recommended) ---
try:
    _fmt = importlib.import_module(".formatter", __name__)
    format_emotions = getattr(_fmt, "format_emotions")
except Exception:
    # Safe fallback: return the dataclass as a dict if formatter not available
    def format_emotions(result):  # type: ignore
        try:
            return result.to_dict()  # type: ignore[attr-defined]
        except Exception:
            return vars(result)

# --- Errors (optional; if not present, provide minimal shims so imports succeed) ---
try:
    _err = importlib.import_module(".errors", __name__)
    InvalidTextError = getattr(_err, "InvalidTextError")
    ServiceUnavailableError = getattr(_err, "ServiceUnavailableError")
except Exception:
    class InvalidTextError(ValueError):  # type: ignore
        pass
    class ServiceUnavailableError(RuntimeError):  # type: ignore
        pass

__all__ = [
    "EmotionResult",
    "emotion_detector",
    "format_emotions",
    "InvalidTextError",
    "ServiceUnavailableError",
]
