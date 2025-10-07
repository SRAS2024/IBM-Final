"""Output formatting helpers with blended-emotion naming for seven cores.

Cores: anger, disgust, fear, joy, sadness, passion, surprise
This module converts raw detector scores into:
- a normalized mixture across seven cores
- an entropy based confidence
- a human friendly blended name
- a single-label 'emotion' (final state) suitable for UI
- a suggested emoji list for UI display
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, List, Tuple

from .detector import EmotionResult

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "passion", "surprise"]

# Canonical pair names for blended_emotion (NOT for final 'emotion' label)
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
    # pair with fear kept only for blended naming, final emotion stays single
    tuple(sorted(["passion", "fear"])): "Aflutter (nervous)",
}

# Triad names are optional. Keep a few illustrative ones (for blended only).
TRIAD_NAMES = {
    tuple(sorted(["anger", "disgust", "fear"])): "Moral outrage",
    tuple(sorted(["anger", "sadness", "fear"])): "Distress",
    tuple(sorted(["joy", "fear", "sadness"])): "Bittersweet anticipation",
    tuple(sorted(["joy", "sadness", "disgust"])): "Embarrassed amusement",
}

# Emoji suggestions keyed by single labels (for final 'emotion')
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

# ---------- Single-state logic (drives the 'emotion' field) ----------

def _single_state_overrides(p: Dict[str, float]) -> str | None:
    """Return a single label when mixture strongly implies a specific state."""
    sad = p["sadness"]; joy = p["joy"]; pas = p["passion"]; fear = p["fear"]
    sup = p["surprise"]; ang = p["anger"]

    # Clear single-core dominance
    ranked = _top_components(p)
    k1, v1 = ranked[0]
    if v1 >= 0.60 and (v1 - ranked[1][1]) >= 0.15:
        return _title(k1)

    # Thematic states
    if sad >= 0.65 and joy <= 0.20:
        return "Mourning"
    if pas >= 0.55 and joy >= 0.25:
        return "In love"
    if sup >= 0.50 and fear >= 0.25:
        return "Shock"
    if sup >= 0.45 and joy >= 0.25:
        return "Awe"  # single label; blended may still say "Delighted surprise"
    if sup >= 0.45 and ang >= 0.25:
        return "Outrage"  # single label; blended can be "Indignant shock"
    if joy >= 0.55 and sad >= 0.25:
        return "Nostalgia"

    return None

def _final_emotion_label(p: Dict[str, float]) -> str:
    """Always return a single emotion label (never a sentence)."""
    single = _single_state_overrides(p)
    if single:
        return single

    # Otherwise choose top component as the single label
    k1, v1 = _top_components(p)[0]
    if v1 < 0.20:
        return "N/A"
    return _title(k1)

# ---------- Blended label (for extra context; may be pair/triad) ----------

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

    # Default to top with qualifier
    return f"{_title(k1)} leaning {_title(k2)}"

# ---------- Emoji selection ----------

def _emoji_for(label: str) -> List[str]:
    if label in EMOJI_SUGGEST:
        return EMOJI_SUGGEST[label]
    # Safe fallbacks for unknown labels
    if label in ("N/A", "Mixed State"):
        return EMOJI_SUGGEST["N/A"]
    # Try core names
    guess = EMOJI_SUGGEST.get(_title(label))
    return guess if guess else ["🤔"]

# ---------- Public formatter ----------

def _round3(x: float) -> float:
    return float(f"{x:.3f}")

def format_emotions(result: EmotionResult) -> Dict[str, object]:
    """Return a dict with seven raw scores and presentation metadata.

    Output keys:
      anger, disgust, fear, joy, sadness, passion, surprise
      dominant_emotion
      blended_emotion     (pair/triad-friendly)
      emotion             (single label, never a sentence)
      confidence          (0..1)
      mixture             (normalized scores, 3dp)
      components          (sorted components for transparency)
      emoji               (1–2 suggested emoji aligned with 'emotion')
    """
    base = asdict(result)

    raw = {k: float(base[k]) for k in EMOTIONS}
    p = _normalize(raw)

    # Slight neutral floor so a tiny blip doesn't dominate noise
    p = {k: max(0.0, v - 0.02) for k, v in p.items()}
    s = sum(p.values()) or 1.0
    p = {k: v / s for k, v in p.items()}

    blended = _blend_name(p)
    final_single = _final_emotion_label(p)
    conf = _confidence(p)

    mixture = {k: _round3(v) for k, v in p.items()}
    components = [(k, _round3(v)) for k, v in _top_components(p)]
    emoji = _emoji_for(final_single)

    base.update(
        {
            "blended_emotion": blended,
            "emotion": final_single,
            "confidence": _round3(conf),
            "mixture": mixture,
            "components": components,
            "emoji": emoji,
        }
    )
    return base
