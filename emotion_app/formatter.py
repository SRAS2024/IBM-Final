"""Output formatting helpers for seven core emotions.

Cores: anger, disgust, fear, joy, sadness, passion, surprise
This module converts raw detector scores into:
- a normalized mixture across seven cores
- an entropy based confidence
- a human friendly blended name
- a single label 'emotion' suitable for UI
- a suggested emoji list for UI display
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, List, Tuple

from .detector import EmotionResult

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "passion", "surprise"]

# Canonical pair names for blended_emotion
PAIR_NAMES = {
    tuple(sorted(["anger", "disgust"])): "Contempt",
    tuple(sorted(["anger", "fear"])): "Outrage",
    tuple(sorted(["fear", "sadness"])): "Anxiety",
    tuple(sorted(["joy", "sadness"])): "Nostalgia",
    tuple(sorted(["joy", "fear"])): "Awe",
    tuple(sorted(["joy", "disgust"])): "Schadenfreude",
    tuple(sorted(["joy", "surprise"])): "Delighted surprise",
    tuple(sorted(["fear", "surprise"])): "Shock",
    tuple(sorted(["anger", "surprise"])): "Indignant shock",
    tuple(sorted(["passion", "joy"])): "In love",
    tuple(sorted(["passion", "fear"])): "Aflutter",
}

TRIAD_NAMES = {
    tuple(sorted(["anger", "disgust", "fear"])): "Moral outrage",
    tuple(sorted(["anger", "sadness", "fear"])): "Distress",
    tuple(sorted(["joy", "fear", "sadness"])): "Bittersweet anticipation",
    tuple(sorted(["joy", "sadness", "disgust"])): "Embarrassed amusement",
}

EMOJI_SUGGEST = {
    "Anger": ["😠"],
    "Disgust": ["🤢"],
    "Fear": ["😨"],
    "Joy": ["😊"],
    "Sadness": ["😢"],
    "Passion": ["😍"],
    "Surprise": ["😮"],
    "Mourning": ["😢"],
    "In love": ["😍"],
    "Awe": ["😮", "✨"],
    "Nostalgia": ["🕰️", "🙂"],
    "Contempt": ["😒"],
    "Outrage": ["😡"],
    "Shock": ["😱"],
    "Mixed State": ["🤔"],
    "N/A": ["🤔"],
}

def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize scores to a probability like mixture without flooring to zero."""
    s = sum(max(0.0, scores.get(k, 0.0)) for k in EMOTIONS)
    if s <= 0:
        # uniform tiny mass so downstream never divides by zero
        return {k: 1.0 / len(EMOTIONS) for k in EMOTIONS}
    return {k: max(0.0, scores.get(k, 0.0)) / s for k in EMOTIONS}

def _entropy(p: Dict[str, float]) -> float:
    eps = 1e-12
    return -sum(pi * math.log(pi + eps) for pi in p.values())

def _confidence(p: Dict[str, float]) -> float:
    h = _entropy(p)
    h_max = math.log(len(EMOTIONS))
    return max(0.0, min(1.0, 1.0 - h / h_max))

def _top_components(p: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(p.items(), key=lambda kv: kv[1], reverse=True)

def _title(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s

# Single state logic for the 'emotion' field
def _single_state_overrides(p: Dict[str, float]) -> str | None:
    sad = p["sadness"]; joy = p["joy"]; pas = p["passion"]; fear = p["fear"]
    sup = p["surprise"]; ang = p["anger"]

    ranked = _top_components(p)
    k1, v1 = ranked[0]
    if v1 >= 0.60 and (v1 - ranked[1][1]) >= 0.15:
        return _title(k1)

    if sad >= 0.65 and joy <= 0.20:
        return "Mourning"
    if pas >= 0.55 and joy >= 0.25:
        return "In love"
    if sup >= 0.50 and fear >= 0.25:
        return "Shock"
    if sup >= 0.45 and joy >= 0.25:
        return "Awe"
    if sup >= 0.45 and ang >= 0.25:
        return "Outrage"
    if joy >= 0.55 and sad >= 0.25:
        return "Nostalgia"

    return None

def _final_emotion_label(p: Dict[str, float]) -> str:
    single = _single_state_overrides(p)
    if single:
        return single
    k1, v1 = _top_components(p)[0]
    if v1 < 0.20:
        return "N/A"
    return _title(k1)

# Blended label for context
def _blend_name(p: Dict[str, float]) -> str:
    ranked = _top_components(p)
    k1, v1 = ranked[0]
    k2, v2 = ranked[1]
    v3 = ranked[2][1]

    if v1 >= 0.60 and (v1 - v2) >= 0.15:
        return _title(k1)

    if (v1 + v2) >= 0.70 and abs(v1 - v2) < 0.20:
        pair = tuple(sorted([k1, k2]))
        return PAIR_NAMES.get(pair, f"{_title(pair[0])} + {_title(pair[1])}")

    if (v1 + v2 + v3) >= 0.85 and v1 < 0.50:
        tri = tuple(sorted([ranked[0][0], ranked[1][0], ranked[2][0]]))
        return TRIAD_NAMES.get(tri, "Mixed State")

    if v1 < 0.35:
        return "N/A"

    return f"{_title(k1)} leaning {_title(k2)}"

def _emoji_for(label: str) -> List[str]:
    if label in EMOJI_SUGGEST:
        return EMOJI_SUGGEST[label]
    guess = EMOJI_SUGGEST.get(_title(label))
    return guess if guess else ["🤔"]

def _round3(x: float) -> float:
    return float(f"{x:.3f}")

def format_emotions(result: EmotionResult) -> Dict[str, object]:
    """Return a dict with seven raw scores and presentation metadata.

    Output keys:
      anger, disgust, fear, joy, sadness, passion, surprise
      dominant_emotion
      blended_emotion
      emotion
      confidence
      mixture
      components
      emoji
    """
    base = asdict(result)

    # Ensure the seven core scores are always present and non negative floats
    raw = {k: float(max(0.0, base.get(k, 0.0))) for k in EMOTIONS}
    p = _normalize(raw)

    blended = _blend_name(p)
    final_single = _final_emotion_label(p)
    conf = _confidence(p)

    mixture = {k: _round3(v) for k, v in p.items()}
    components = [{"name": k, "weight": _round3(v)} for k, v in _top_components(p)]
    emoji = _emoji_for(final_single)

    # Update base while keeping dominant_emotion from the detector
    base.update(
        {
            "anger": raw["anger"],
            "disgust": raw["disgust"],
            "fear": raw["fear"],
            "joy": raw["joy"],
            "sadness": raw["sadness"],
            "passion": raw["passion"],
            "surprise": raw["surprise"],
            "blended_emotion": blended,
            "emotion": final_single,
            "confidence": _round3(conf),
            "mixture": mixture,
            "components": components,
            "emoji": emoji,
        }
    )
    return base
