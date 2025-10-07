"""Emotion detection using Watson NLU with a graceful local fallback."""
from __future__ import annotations

import os
import re
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
        return {
            "anger": float(doc_emotions.get("anger", 0.0)),
            "disgust": float(doc_emotions.get("disgust", 0.0)),
            "fear": float(doc_emotions.get("fear", 0.0)),
            "joy": float(doc_emotions.get("joy", 0.0)),
            "sadness": float(doc_emotions.get("sadness", 0.0)),
        }
    except Exception as exc:  # pragma: no cover
        raise ServiceUnavailableError(f"Watson call failed: {exc}") from exc


_WORDS = {
    "joy": [
        "love", "happy", "joy", "excited", "grateful", "great", "wonderful",
        "amazing", "glad", "smile", "laugh", "proud", "delight", "cheerful"
    ],
    "sadness": [
        "sad", "down", "lonely", "depressed", "cry", "tearful", "unhappy",
        "miserable", "heartbroken", "sorrow"
    ],
    "anger": [
        "angry", "mad", "furious", "rage", "irritated", "annoyed", "upset",
        "hate", "outraged", "resentful"
    ],
    "fear": [
        "scared", "afraid", "fear", "terrified", "anxious", "worried",
        "panic", "nervous", "phobia"
    ],
    "disgust": [
        "disgust", "gross", "nasty", "revolting", "repulsed", "sickened"
    ],
}

_NEGATIONS = {"not", "no", "never", "hardly", "barely", "without"}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _fallback_model(text: str) -> Dict[str, float]:
    """Simple heuristic for offline development and tests with basic negation handling."""
    tokens = _tokenize(text)
    if not tokens:
        return {k: 0.0 for k in ["anger", "disgust", "fear", "joy", "sadness"]}

    # Count keyword hits with a small boost for emphasis
    scores: Dict[str, float] = {k: 0.0 for k in _WORDS}
    for i, tok in enumerate(tokens):
        for emo, lex in _WORDS.items():
            if tok in lex:
                weight = 1.0

                # Negation window
                window = tokens[max(0, i - 3):i]
                if any(w in _NEGATIONS for w in window):
                    weight *= -0.8  # invert toward opposite

                scores[emo] += weight

    # Map negative weights from negation into opposite emotions
    if scores["joy"] < 0:
        scores["sadness"] += abs(scores["joy"]) * 0.8
        scores["joy"] = 0.0
    if scores["fear"] < 0:
        scores["anger"] += abs(scores["fear"]) * 0.6
        scores["fear"] = 0.0
    if scores["anger"] < 0:
        scores["fear"] += abs(scores["anger"]) * 0.6
        scores["anger"] = 0.0
    if scores["disgust"] < 0:
        scores["joy"] += abs(scores["disgust"]) * 0.4
        scores["disgust"] = 0.0
    if scores["sadness"] < 0:
        scores["joy"] += abs(scores["sadness"]) * 0.6
        scores["sadness"] = 0.0

    # Normalize by token count and clamp to [0, 1]
    norm = max(len(tokens), 1)
    for k in scores:
        scores[k] = max(0.0, min(scores[k] / norm * 3.0, 1.0))  # gentle scaling

    return {
        "anger": float(scores["anger"]),
        "disgust": float(scores["disgust"]),
        "fear": float(scores["fear"]),
        "joy": float(scores["joy"]),
        "sadness": float(scores["sadness"]),
    }


def _choose_dominant(scores: Dict[str, float]) -> str:
    # If all scores are near zero, report N/A to avoid defaulting to anger
    if sum(scores.values()) < 0.001 or max(scores.values()) < 0.05:
        return "N/A"
    return max(scores, key=scores.get)


def emotion_detector(text: str) -> EmotionResult:
    """Detect emotions in text and return a structured result."""
    if text is None or not str(text).strip():
        raise InvalidTextError("Input text is required.")

    if _has_watson_credentials():
        scores = _call_watson(text)
    else:
        scores = _fallback_model(text)

    dominant = _choose_dominant(scores)
    return EmotionResult(
        anger=scores.get("anger", 0.0),
        disgust=scores.get("disgust", 0.0),
        fear=scores.get("fear", 0.0),
        joy=scores.get("joy", 0.0),
        sadness=scores.get("sadness", 0.0),
        dominant_emotion=dominant,
    )
