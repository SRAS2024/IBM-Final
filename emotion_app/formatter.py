# -*- coding: utf-8 -*-
"""Output formatting helpers for seven core emotions with rich final labels.

Cores: anger, disgust, fear, joy, sadness, passion, surprise

This module converts detector scores into:
- a normalized seven-way mixture suitable for a bar chart
- an entropy-based confidence
- a contextual blended name from pair or triad
- a single rich 'emotion' label from prototype space and overrides
- separate emojis for dominant core and rich emotion
- a concise rationale string for UI captions
- backward-compatible fields for existing callers

Compatible with detector.py (contrast-aware / near-tie dominance).
"""
from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Dict, List, Tuple, Optional, Any

from .detector import EmotionResult

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "passion", "surprise"]
_IDX = {k: i for i, k in enumerate(EMOTIONS)}

# Canonical names for top blends
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

# Rich single label prototypes over the seven cores
PROTOTYPES: Dict[str, List[float]] = {
    # Positives
    "Joyful":               [0.02, 0.00, 0.02, 1.00, 0.00, 0.05, 0.10],
    "Satisfied":            [0.00, 0.00, 0.00, 0.80, 0.05, 0.10, 0.05],
    "Relief":               [0.00, 0.00, 0.10, 0.70, 0.10, 0.00, 0.05],
    "Calm":                 [0.00, 0.00, 0.00, 0.55, 0.05, 0.00, 0.00],

    # Romantic and attachment
    "In love":              [0.00, 0.00, 0.00, 0.45, 0.00, 1.00, 0.15],
    "Infatuated":           [0.00, 0.00, 0.05, 0.35, 0.00, 0.95, 0.25],
    "Longing":              [0.00, 0.00, 0.10, 0.10, 0.55, 0.70, 0.10],
    "Committed":            [0.00, 0.00, 0.00, 0.40, 0.05, 0.85, 0.05],

    # Surprise blends
    "Awe":                  [0.00, 0.00, 0.20, 0.60, 0.10, 0.10, 0.90],
    "Shocked":              [0.10, 0.00, 0.55, 0.05, 0.05, 0.00, 1.00],
    "Delighted surprise":   [0.00, 0.00, 0.10, 0.70, 0.00, 0.10, 0.90],

    # Negatives
    "Angry":                [1.00, 0.10, 0.10, 0.00, 0.05, 0.00, 0.10],
    "Disgusted":            [0.30, 1.00, 0.05, 0.00, 0.10, 0.00, 0.05],
    "Appalled":             [0.20, 0.90, 0.20, 0.00, 0.10, 0.00, 0.30],
    "Contempt":             [0.70, 0.80, 0.10, 0.00, 0.10, 0.00, 0.10],
    "Outrage":              [0.95, 0.15, 0.35, 0.00, 0.15, 0.00, 0.40],

    # Anxiety and fear spectrum
    "Anxious":              [0.05, 0.00, 1.00, 0.00, 0.10, 0.00, 0.20],
    "Apprehensive":         [0.05, 0.00, 0.85, 0.10, 0.10, 0.00, 0.15],
    "Uneasy":               [0.05, 0.05, 0.70, 0.10, 0.10, 0.00, 0.10],

    # Sadness spectrum
    "Sad":                  [0.05, 0.00, 0.10, 0.00, 1.00, 0.00, 0.00],
    "Mourning":             [0.05, 0.00, 0.10, 0.00, 0.95, 0.00, 0.00],
    "Heartbroken":          [0.25, 0.00, 0.10, 0.05, 0.95, 0.40, 0.10],
    "Melancholy":           [0.00, 0.00, 0.05, 0.10, 0.85, 0.00, 0.05],
    "Nostalgia":            [0.00, 0.00, 0.05, 0.45, 0.45, 0.05, 0.10],

    # Mixed states
    "Bittersweet":          [0.05, 0.00, 0.10, 0.50, 0.55, 0.10, 0.20],
    "Indignant shock":      [0.85, 0.00, 0.25, 0.05, 0.10, 0.00, 0.80],
    "Moral outrage":        [0.90, 0.70, 0.50, 0.00, 0.25, 0.00, 0.20],
    "Schadenfreude":        [0.15, 0.60, 0.05, 0.55, 0.10, 0.00, 0.10],
    "Embarrassed amusement":[0.10, 0.35, 0.10, 0.55, 0.35, 0.00, 0.10],
}

# Suggested emoji keyed by label
EMOJI_SUGGEST = {
    # Rich labels
    "Angry": ["😠"], "Disgusted": ["🤢"], "Anxious": ["😨"], "Joyful": ["😊"],
    "Sad": ["😢"], "In love": ["😍"], "Shocked": ["😱"], "Awe": ["😮", "✨"],
    "Nostalgia": ["🕰️", "🙂"], "Contempt": ["😒"], "Outrage": ["😡"],
    "Bittersweet": ["🥲"], "Mourning": ["🖤"], "Heartbroken": ["💔"],
    "Infatuated": ["🥰"], "Committed": ["💍"], "Relief": ["😮\u200d💨"], "Calm": ["😌"],
    "Appalled": ["😧"], "Uneasy": ["😬"], "Apprehensive": ["😟"],
    "Delighted surprise": ["🤩"], "Indignant shock": ["😤", "😳"],
    "Moral outrage": ["😤"], "Schadenfreude": ["😏"], "Embarrassed amusement": ["😅"],
    "Melancholy": ["🎻"],

    # Core labels
    "Anger": ["😠"],
    "Disgust": ["🤢"],
    "Fear": ["😨"],
    "Joy": ["😊"],
    "Sadness": ["😢"],
    "Passion": ["😍"],
    "Surprise": ["😮"],

    # Fallback
    "N/A": ["🤔"],
}

# ------------------------- numeric helpers -------------------------

def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, scores.get(k, 0.0)) for k in EMOTIONS)
    if s <= 0:
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

def _round3(x: float) -> float:
    return float(f"{x:.3f}")

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _norm(v: List[float]) -> float:
    n = math.sqrt(sum(x * x for x in v))
    return n if n > 0 else 1.0

def _cosine(a: List[float], b: List[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))

# ------------------------- single state logic -------------------------

def _single_state_overrides(p: Dict[str, float]) -> Optional[str]:
    sad = p["sadness"]; joy = p["joy"]; pas = p["passion"]; fear = p["fear"]
    sup = p["surprise"]; ang = p["anger"]

    ranked = _top_components(p)
    k1, v1 = ranked[0]
    if v1 >= 0.60 and (v1 - ranked[1][1]) >= 0.15:
        return _title(k1)

    if sad >= 0.65 and joy <= 0.20:
        return "Mourning"
    # Passion plus joy override for clear romantic language
    if (pas >= 0.50 and joy >= 0.22) or ((pas + joy) >= 0.70 and abs(pas - joy) <= 0.08):
        return "In love"
    if pas >= 0.65 and sad >= 0.25 and joy <= 0.25:
        return "Longing"
    if sup >= 0.50 and fear >= 0.25:
        return "Shocked"
    if sup >= 0.45 and joy >= 0.25:
        return "Awe"
    if sup >= 0.45 and ang >= 0.25:
        return "Outrage"
    if joy >= 0.55 and sad >= 0.25:
        return "Nostalgia"

    return None

# ------------------------- blended label for context -------------------------

def _blend_name(p: Dict[str, float]) -> str:
    ranked = _top_components(p)
    k1, v1 = ranked[0]
    k2, v2 = ranked[1]
    v3 = ranked[2][1]

    # If one emotion clearly dominates, use it as a contextual label
    if v1 >= 0.60 and (v1 - v2) >= 0.15:
        return _title(k1)

    # Balanced pair
    if (v1 + v2) >= 0.70 and abs(v1 - v2) < 0.20:
        pair = tuple(sorted([k1, k2]))
        return PAIR_NAMES.get(pair, f"{_title(pair[0])} + {_title(pair[1])}")

    # Triad
    if (v1 + v2 + v3) >= 0.85 and v1 < 0.50:
        tri = tuple(sorted([ranked[0][0], ranked[1][0], ranked[2][0]]))
        return TRIAD_NAMES.get(tri, "Mixed State")

    # Low signal
    if v1 < 0.35:
        return "N/A"

    return f"{_title(k1)} leaning {_title(k2)}"

# ------------------------- rich label selection -------------------------

def _vector_from_p(p: Dict[str, float]) -> List[float]:
    return [p[k] for k in EMOTIONS]

def _best_prototype_label(p: Dict[str, float]) -> Tuple[str, float]:
    v = _vector_from_p(p)
    best_label = "N/A"
    best_sim = -1.0
    for label, proto in PROTOTYPES.items():
        sim = _cosine(v, proto)
        if sim > best_sim:
            best_label, best_sim = label, sim
    return best_label, best_sim

def _final_emotion_label(p: Dict[str, float]) -> str:
    # Deterministic overrides first
    single = _single_state_overrides(p)
    if single:
        return single

    # If top two are Passion and Joy and reasonably strong, call it In love
    ranked = _top_components(p)
    (k1, v1), (k2, v2) = ranked[0], ranked[1]
    if {"passion", "joy"} == {k1, k2} and (v1 + v2) >= 0.60 and abs(v1 - v2) <= 0.20:
        return "In love"

    # Prototype choice
    conf = _confidence(p)
    if v1 >= 0.22 and conf >= 0.15:
        label, sim = _best_prototype_label(p)
        if sim >= 0.72:
            return label

    # Mirror dominant if nothing else qualifies
    if v1 < 0.20:
        return "N/A"
    return _title(k1)

# ------------------------- emoji selection -------------------------

def _emoji_for(label: str) -> List[str]:
    if label in EMOJI_SUGGEST:
        return EMOJI_SUGGEST[label]
    guess = EMOJI_SUGGEST.get(_title(label))
    return guess if guess else ["🤔"]

def _emoji_for_core(core_lower: str) -> List[str]:
    return _emoji_for(_title(core_lower))

# ------------------------- present component filtering -------------------------

def _present_subset(p: Dict[str, float], eps: float = 0.03) -> Dict[str, float]:
    """Return only components with weight above eps. Sorted high to low."""
    items = [(k, v) for k, v in p.items() if v >= eps]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return {k: _round3(v) for k, v in items}

# ------------------------- input normalization -------------------------

def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Accept EmotionResult dataclass or a mapping with compatible keys."""
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return dict(result)
    # Last resort: try attribute access (duck-typing)
    out = {}
    for k in ["anger", "disgust", "fear", "joy", "sadness", "passion", "surprise", "dominant_emotion"]:
        out[k] = float(getattr(result, k)) if hasattr(result, k) else 0.0
    if "dominant_emotion" not in out:
        out["dominant_emotion"] = ""
    return out

def _ensure_score_keys(base: Dict[str, Any]) -> Dict[str, Any]:
    for k in EMOTIONS:
        base[k] = float(base.get(k, 0.0) or 0.0)
    base["dominant_emotion"] = (base.get("dominant_emotion") or "") if isinstance(base.get("dominant_emotion"), str) else ""
    return base

def _fallback_dominant_if_missing(raw: Dict[str, float], existing: str) -> str:
    """If detector didn't provide dominant_emotion, choose the max of the normalized mix."""
    existing = existing or ""
    if existing:
        return existing
    p = _normalize(raw)
    top = max(p.items(), key=lambda kv: kv[1])[0] if p else "N/A"
    return top

# ------------------------- rationale -------------------------

def _rationale(p: Dict[str, float], dominant_lower: str, emotion_label: str, blended_label: str) -> str:
    ranked = _top_components(p)[:3]
    parts = [f"{_title(k)} {_round3(v)}" for k, v in ranked]
    # Explain collapse rule expectation: UI may render one emoji when labels match
    if _title(dominant_lower) == emotion_label:
        why = f"Dominant and rich emotion match as {_title(dominant_lower)}."
    else:
        why = f"Dominant is {_title(dominant_lower)}. Rich label reflects blend as {emotion_label}."
    return f"Top mix: {', '.join(parts)}. {why} Blended context: {blended_label}."

# ------------------------- public formatter -------------------------

def format_emotions(result: EmotionResult | Dict[str, Any]) -> Dict[str, object]:
    """Return a dict with seven raw scores and presentation metadata.

    Output keys:
      anger, disgust, fear, joy, sadness, passion, surprise
      dominant_emotion
      blended_emotion
      emotion
      confidence
      mixture
      present
      components
      emoji_emotion
      emoji_dominant
      same_label
      rationale
      emoji          # alias of emoji_emotion for backward compatibility
      emoji_primary  # first item of emoji_emotion for backward compatibility
    """
    base = _result_to_dict(result)
    base = _ensure_score_keys(base)

    # Normalize to a probability-like mixture for charting
    raw = {k: float(max(0.0, base.get(k, 0.0))) for k in EMOTIONS}
    p = _normalize(raw)

    # Labels
    blended = _blend_name(p)
    final_single = _final_emotion_label(p)
    conf = _confidence(p)

    # Dominant core from detector is lowercase; fall back if missing
    dominant_lower = _fallback_dominant_if_missing(raw, base.get("dominant_emotion", "") or "")
    dominant_title = _title(dominant_lower) if dominant_lower else "N/A"

    # Emojis
    emoji_emotion = _emoji_for(final_single)
    emoji_dominant = _emoji_for_core(dominant_lower if dominant_lower else "N/A")

    # Backward compatibility and chart data
    mixture = {k: _round3(v) for k, v in p.items()}
    components = [{"name": k, "weight": _round3(v)} for k, v in _top_components(p)]
    present = _present_subset(p, eps=0.03)

    same_label = dominant_title == final_single
    rationale = _rationale(p, dominant_lower, final_single, blended)

    # Assemble output
    base.update(
        {
            # Scores
            "anger": raw["anger"],
            "disgust": raw["disgust"],
            "fear": raw["fear"],
            "joy": raw["joy"],
            "sadness": raw["sadness"],
            "passion": raw["passion"],
            "surprise": raw["surprise"],

            # Labels
            "dominant_emotion": dominant_lower or "N/A",
            "blended_emotion": blended,
            "emotion": final_single,                 # rich single label
            "confidence": _round3(conf),

            # Charting
            "mixture": mixture,                      # normalized seven-way for bars
            "present": present,                      # nontrivial components only
            "components": components,                # sorted list for debugging or tables

            # Emojis
            "emoji_emotion": emoji_emotion,          # for rich label
            "emoji_dominant": emoji_dominant,        # for dominant core
            "same_label": same_label,                # UI can collapse to one emoji if True

            # Narrative
            "rationale": rationale,

            # Backward compatibility
            "emoji": emoji_emotion,
            "emoji_primary": emoji_emotion[0] if emoji_emotion else "🤔",
        }
    )
    return base
