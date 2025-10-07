"""Output formatting helpers with blended-emotion naming."""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, List, Tuple

from .detector import EmotionResult

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness"]

# Human-friendly blend names for common pairs and triads
PAIR_NAMES = {
    ("anger", "disgust"): "Contempt",
    ("anger", "sadness"): "Envy",
    ("anger", "fear"): "Outrage",
    ("fear", "sadness"): "Anxiety",
    ("joy", "sadness"): "Nostalgia",
    ("joy", "fear"): "Awe",
    ("joy", "disgust"): "Schadenfreude",
}

TRIAD_NAMES = {
    tuple(sorted(["anger", "disgust", "fear"])): "Moral Outrage",
    tuple(sorted(["anger", "sadness", "fear"])): "Distress",
    tuple(sorted(["joy", "fear", "sadness"])): "Bittersweet Anticipation",
    tuple(sorted(["joy", "sadness", "disgust"])): "Embarrassed Amusement",
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
    return f"{_title(k1)} leaning { _title(k2) }"

def _round3(x: float) -> float:
    return float(f"{x:.3f}")

def format_emotions(result: EmotionResult) -> Dict[str, object]:
    """Return a dict in the expected shape with blended emotion metadata.

    Output keys:
      anger, disgust, fear, joy, sadness          raw scores from the detector
      dominant_emotion                            original dominant choice
      blended_emotion                             named blend computed from normalized scores
      confidence                                  0.0 to 1.0 based on distribution entropy
      mixture                                     normalized five-way mixture rounded to 3 decimals
      components                                  top components in order for transparency
    """
    base = asdict(result)

    raw = {k: float(base[k]) for k in EMOTIONS}
    p = _normalize(raw)
    name = _blend_name(p)
    conf = _confidence(p)

    mixture = {k: _round3(v) for k, v in p.items()}
    components = [(k, _round3(v)) for k, v in _top_components(p)]

    base.update(
        {
            "blended_emotion": name,
            "confidence": _round3(conf),
            "mixture": mixture,
            "components": components,
        }
    )
    return base
