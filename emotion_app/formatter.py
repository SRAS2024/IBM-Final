"""Output formatting helpers with blended-emotion naming for seven cores.

Cores: anger, disgust, fear, joy, sadness, passion, surprise
This module converts raw detector scores into:
- a normalized mixture across seven cores
- an entropy based confidence
- a human friendly blended name
- a final two emotion label under the key "emotion"
- a suggested emoji or emoji pair for UI display
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, List, Tuple

from .detector import EmotionResult

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "passion", "surprise"]

# Canonical pair names for common blends
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
    tuple(sorted(["passion", "fear"])): "In love but afraid",
}

# Triad names are optional. Keep a few illustrative ones.
TRIAD_NAMES = {
    tuple(sorted(["anger", "disgust", "fear"])): "Moral outrage",
    tuple(sorted(["anger", "sadness", "fear"])): "Distress",
    tuple(sorted(["joy", "fear", "sadness"])): "Bittersweet anticipation",
    tuple(sorted(["joy", "sadness", "disgust"])): "Embarrassed amusement",
}

# Emoji suggestions. UI can join these or use only the first.
EMOJI_SUGGEST = {
    "Anger": ["😠"],
    "Disgust": ["🤢"],
    "Fear": ["😨"],
    "Joy": ["😊"],
    "Sadness": ["😢"],
    "Passion": ["😍"],
    "Surprise": ["😮"],
    "Contempt": ["😒"],
    "Outrage": ["😡", "😠"],
    "Anxiety": ["😰"],
    "Nostalgia": ["🕰️", "🙂"],
    "Awe": ["😮", "✨"],
    "Schadenfreude": ["😏"],
    "Delighted surprise": ["😮", "😊"],
    "Shock": ["😱"],
    "Indignant shock": ["😠", "😮"],
    "In love": ["😍"],
    "In love but afraid": ["😨", "😍"],
    "Mourning": ["😢"],
    "Happy but mourning": ["😢", "🙂"],
    "Mixed State": ["🤔"],
    "N/A": ["🤔"],
}

def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    s = sum(scores.get(k, 0.0) for k in EMOTIONS)
    if s <= 0:
        return {k: 0.0 for k in EMOTIONS}
    return {k: scores.get(k, 0.0) / s for k in EMOTIONS}

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

def _single_state_overrides(p: Dict[str, float]) -> str | None:
    """Return a single state label when mixture strongly implies a specific named state."""
    sad = p["sadness"]
    joy = p["joy"]
    pas = p["passion"]
    fear = p["fear"]

    # Mourning when sadness dominates and joy is low
    if sad >= 0.65 and joy <= 0.20:
        return "Mourning"

    # In love when passion is high with supportive joy and low conflict
    if pas >= 0.55 and joy >= 0.25 and fear < 0.35 and sad < 0.40:
        return "In love"

    # In love but afraid when passion and fear co-exist
    if pas >= 0.45 and fear >= 0.25:
        return "In love but afraid"

    # Shock like states driven by surprise
    sup = p["surprise"]
    ang = p["anger"]
    if sup >= 0.45 and fear >= 0.25:
        return "Shock"
    if sup >= 0.45 and joy >= 0.25:
        return "Delighted surprise"
    if sup >= 0.45 and ang >= 0.25:
        return "Indignant shock"

    # Happy but mourning when sadness is high but joy is notable
    if sad >= 0.55 and joy >= 0.20:
        return "Happy but mourning"

    return None

def _blend_name(p: Dict[str, float]) -> str:
    ranked = _top_components(p)
    k1, v1 = ranked[0]
    k2, v2 = ranked[1]
    v3 = ranked[2][1]

    # Strong single emotion
    if v1 >= 0.60 and (v1 - v2) >= 0.15:
        return _title(k1)

    # Common pair blends
    if (v1 + v2) >= 0.70 and (v1 - v2) < 0.20:
        pair = tuple(sorted([k1, k2]))
        if pair in PAIR_NAMES:
            return PAIR_NAMES[pair]
        return f"{_title(pair[0])} + {_title(pair[1])}"

    # Triad when evidence is spread but focused
    if (v1 + v2 + v3) >= 0.85 and v1 < 0.50:
        tri = tuple(sorted([ranked[0][0], ranked[1][0], ranked[2][0]]))
        if tri in TRIAD_NAMES:
            return TRIAD_NAMES[tri]
        return "Mixed State"

    # Weak or flat evidence
    if v1 < 0.35:
        return "N/A"

    # Default to top emotion with qualifier
    return f"{_title(k1)} leaning {_title(k2)}"

def _two_emotion_label(p: Dict[str, float], blended: str) -> str:
    """Final two emotion label under the key 'emotion'."""
    # First respect high precision single state overrides
    single = _single_state_overrides(p)
    if single:
        return single

    # Otherwise name by the top two
    ranked = _top_components(p)
    k1, v1 = ranked[0]
    k2, v2 = ranked[1]

    # Try canonical pair name if available
    pair = tuple(sorted([k1, k2]))
    if pair in PAIR_NAMES:
        return PAIR_NAMES[pair]

    # If evidence is weak
    if v1 < 0.35:
        return "N/A"

    # Generic two way label
    return f"{_title(k1)} and {_title(k2)}"

def _emoji_for(label: str, dominant: str) -> List[str]:
    """Choose one or two emoji that reflect the final emotion, ordered by logic:
    if fear is involved with something positive, show fear first."""
    # Direct mapping when we have it
    if label in EMOJI_SUGGEST:
        return EMOJI_SUGGEST[label]

    # Fallbacks built from components in the label text
    l = label.lower()
    if "love" in l and "afraid" in l:
        return ["😨", "😍"]
    if "love" in l:
        return ["😍"]
    if "mourning" in l:
        return ["😢", "🙂"] if "happy" in l else ["😢"]
    if "shock" in l:
        return ["😱"]
    if "surprise" in l and "delighted" in l:
        return ["😮", "😊"]
    if dominant.capitalize() in EMOJI_SUGGEST:
        return EMOJI_SUGGEST[dominant.capitalize()]
    return ["🤔"]

def _round3(x: float) -> float:
    return float(f"{x:.3f}")

def format_emotions(result: EmotionResult) -> Dict[str, object]:
    """Return a dict with seven raw scores and blended metadata.

    Output keys:
      anger, disgust, fear, joy, sadness, passion, surprise
      dominant_emotion
      blended_emotion           broader human friendly name from the seven way mix
      emotion                   final two emotion label or precise single state
      confidence                0.0 to 1.0 based on distribution entropy
      mixture                   normalized seven way mixture rounded to 3 decimals
      components                top components in order for transparency
      emoji                     suggested emoji or pair for UI
    """
    base = asdict(result)

    raw = {k: float(base[k]) for k in EMOTIONS}
    p = _normalize(raw)

    # Slight neutral floor so a single tiny hit does not dominate
    p = {k: max(0.0, v - 0.02) for k, v in p.items()}
    s = sum(p.values()) or 1.0
    p = {k: v / s for k, v in p.items()}

    name = _blend_name(p)
    final_label = _two_emotion_label(p, name)
    conf = _confidence(p)

    mixture = {k: _round3(v) for k, v in p.items()}
    components = [(k, _round3(v)) for k, v in _top_components(p)]
    emoji = _emoji_for(final_label, base.get("dominant_emotion", ""))

    base.update(
        {
            "blended_emotion": name,
            "emotion": final_label,
            "confidence": _round3(conf),
            "mixture": mixture,
            "components": components,
            "emoji": emoji,
        }
    )
    return base
