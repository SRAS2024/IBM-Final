"""Emotion detection using Watson NLU with a graceful local fallback."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

from .errors import InvalidTextError, ServiceUnavailableError

try:
    # Import lazily to avoid hard dependency when credentials are not provided
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
except Exception:  # pragma: no cover
    NaturalLanguageUnderstandingV1 = None  # type: ignore
    IAMAuthenticator = None  # type: ignore
    Features = None  # type: ignore
    EmotionOptions = None  # type: ignore


@dataclass
class EmotionResult:
    anger: float
    disgust: float
    fear: float
    joy: float
    sadness: float
    dominant_emotion: str


def _has_watson_credentials() -> bool:
    return bool(os.getenv("WATSON_NLU_APIKEY")) and bool(os.getenv("WATSON_NLU_URL"))


def _call_watson(text: str) -> Dict[str, float]:
    """Call IBM Watson NLU emotion endpoint and return emotion scores."""
    if NaturalLanguageUnderstandingV1 is None:
        raise ServiceUnavailableError("Watson SDK not available.")

    apikey = os.getenv("WATSON_NLU_APIKEY")
    url = os.getenv("WATSON_NLU_URL")

    if not apikey or not url:
        raise ServiceUnavailableError("Watson credentials missing.")

    try:
        authenticator = IAMAuthenticator(apikey)
        nlu = NaturalLanguageUnderstandingV1(version="2022-04-07", authenticator=authenticator)
        nlu.set_service_url(url)

        response = nlu.analyze(
            text=text,
            features=Features(emotion=EmotionOptions(document=True)),
        ).get_result()

        doc_emotions = response["emotion"]["document"]["emotion"]
        # Ensure consistent keys and floats
        return {
            "anger": float(doc_emotions.get("anger", 0.0)),
            "disgust": float(doc_emotions.get("disgust", 0.0)),
            "fear": float(doc_emotions.get("fear", 0.0)),
            "joy": float(doc_emotions.get("joy", 0.0)),
            "sadness": float(doc_emotions.get("sadness", 0.0)),
        }
    except Exception as exc:  # pragma: no cover
        raise ServiceUnavailableError(f"Watson call failed: {exc}") from exc


def _fallback_model(text: str) -> Dict[str, float]:
    """Very simple heuristic for offline development and tests."""
    text_l = text.lower()
    joy_words = ["love", "great", "happy", "joy", "wonderful", "amazing"]
    sad_words = ["sad", "lonely", "down", "depressed", "cry"]
    anger_words = ["angry", "mad", "furious", "rage"]
    fear_words = ["scared", "afraid", "fear", "terrified"]
    disgust_words = ["disgust", "gross", "nasty"]

    def score(words):
        return sum(text_l.count(w) for w in words) / max(len(text_l.split()), 1)

    scores = {
        "joy": min(score(joy_words), 1.0),
        "sadness": min(score(sad_words), 1.0),
        "anger": min(score(anger_words), 1.0),
        "fear": min(score(fear_words), 1.0),
        "disgust": min(score(disgust_words), 1.0),
    }
    return {
        "anger": float(scores["anger"]),
        "disgust": float(scores["disgust"]),
        "fear": float(scores["fear"]),
        "joy": float(scores["joy"]),
        "sadness": float(scores["sadness"]),
    }


def emotion_detector(text: str) -> EmotionResult:
    """Detect emotions in text and return a structured result.

    Raises InvalidTextError when text is empty or only whitespace.
    May raise ServiceUnavailableError if Watson service fails when credentials are set.
    """
    if text is None or not str(text).strip():
        raise InvalidTextError("Input text is required.")

    if _has_watson_credentials():
        scores = _call_watson(text)
    else:
        scores = _fallback_model(text)

    dominant = max(scores, key=scores.get) if scores else ""
    return EmotionResult(
        anger=scores.get("anger", 0.0),
        disgust=scores.get("disgust", 0.0),
        fear=scores.get("fear", 0.0),
        joy=scores.get("joy", 0.0),
        sadness=scores.get("sadness", 0.0),
        dominant_emotion=dominant,
    )
