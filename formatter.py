"""Output formatting helpers."""
from dataclasses import asdict
from typing import Dict

from .detector import EmotionResult


def format_emotions(result: EmotionResult) -> Dict[str, object]:
    """Return a dict in the expected shape for the assignment.

    Example shape:
    {
        "anger": 0.01,
        "disgust": 0.02,
        "fear": 0.03,
        "joy": 0.90,
        "sadness": 0.04,
        "dominant_emotion": "joy",
    }
    """
    return asdict(result)
