 package interface for emotion_app."""
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
