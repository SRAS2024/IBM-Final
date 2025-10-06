"""Public package interface for emotion_app."""
from .detector import emotion_detector, EmotionResult
from .formatter import format_emotions
from .errors import InvalidTextError, ServiceUnavailableError

__all__ = [
    "emotion_detector",
    "EmotionResult",
    "format_emotions",
    "InvalidTextError",
    "ServiceUnavailableError",
]
Why this works
Linux is case sensitive. A folder named Final will not be found by import Final if the real name differs even slightly.
By using a lowercase package emotion_app at the project root, Python will always find your modules because the repo root is on PYTHONPATH when Railway starts your app.
Tests
Rename your test files to test_detector.py and test_server.py if they are not already.
In tests, import from emotion_app:
from emotion_app import emotion_detector, InvalidTextError
from server import app
