"""High fidelity local emotion detector.

Design goals
------------
1) Work fully offline with no external services.
2) Handle short phrases, partial sentences, typos, and emoji.
3) Consider common linguistic phenomena:
   - negation scope
   - intensifiers and dampeners
   - contrastive pivots such as "but" and "however"
   - emphasis from capitalization, repetition, and punctuation
   - basic sarcasm cues such as "yeah right", "as if"
   - stance from first person pronouns
4) Provide stable scores in [0, 1] and a sensible dominant emotion.
5) Keep the public API the same as before.

If IBM Watson NLU credentials are supplied through environment variables,
the Watson pathway is used automatically. Otherwise this file provides
a robust rule based fallback.

Public API
----------
emotion_detector(text: str) -> EmotionResult
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .errors import InvalidTextError, ServiceUnavailableError

# Optional Watson imports kept for completeness. If not configured, we use the fallback.
try:  # pragma: no cover
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


# ---------------------------------------------------------------------------
# Watson pathway
# ---------------------------------------------------------------------------

def _has_watson_credentials() -> bool:
    return bool(os.getenv("WATSON_NLU_APIKEY")) and bool(os.getenv("WATSON_NLU_URL"))


def _call_watson(text: str) -> Dict[str, float]:
    """Call IBM Watson NLU if credentials are present."""
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
        result = nlu.analyze(
            text=text,
            features=Features(emotion=EmotionOptions(document=True)),
        ).get_result()
        e = result["emotion"]["document"]["emotion"]
        return {
            "anger": float(e.get("anger", 0.0)),
            "disgust": float(e.get("disgust", 0.0)),
            "fear": float(e.get("fear", 0.0)),
            "joy": float(e.get("joy", 0.0)),
            "sadness": float(e.get("sadness", 0.0)),
        }
    except Exception as exc:  # pragma: no cover
        raise ServiceUnavailableError(f"Watson call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Enhanced rule based fallback
# ---------------------------------------------------------------------------

# Core lexicons. Compact but broad. All lowercased stems.
JOY = {
    "love", "loved", "loving", "like", "liked", "likes",
    "happy", "joy", "joyful", "cheer", "cheerful", "content", "contented",
    "excite", "excited", "exciting", "glad", "delight", "delighted", "delightful",
    "pleased", "satisfy", "satisfied", "awesome", "amazing", "wonderful", "fantastic",
    "great", "good", "grateful", "blessed", "peaceful", "calm", "serene", "hopeful",
    "win", "won", "success", "proud", "smile", "laugh", "yay", "woohoo", "hurray",
    "enjoy", "enjoyed", "enjoying", "fun", "cute", "beautiful", "brilliant",
}
SADNESS = {
    "sad", "sadden", "saddened", "down", "depress", "depressed", "depressing",
    "cry", "cried", "crying", "tearful", "teary", "lonely", "alone", "miserable",
    "heartbroken", "sorrow", "grief", "blue", "hopeless", "helpless", "gloomy",
    "lost", "regret", "regretful", "sorry", "miss", "missing", "tired", "drained",
}
ANGER = {
    "angry", "anger", "mad", "furious", "rage", "raging", "irritated", "annoyed",
    "annoying", "upset", "hate", "hated", "hates", "hating", "outraged", "livid",
    "resentful", "hostile", "fume", "yell", "shout", "scream", "screaming",
    "frustrated", "frustrating", "infuriating", "disrespect", "insult", "insulted",
}
FEAR = {
    "scare", "scared", "afraid", "fear", "fearful", "terrified", "terrify",
    "anxious", "anxiety", "worry", "worried", "worrying", "panic", "panicked",
    "nervous", "phobia", "frighten", "frightened", "tense", "uneasy", "alarmed",
    "concerned", "concern", "dread", "spook", "shaky", "shaking",
}
DISGUST = {
    "disgust", "disgusted", "gross", "nasty", "revolting", "repulsed", "repulsive",
    "sicken", "sickened", "vile", "filthy", "dirty", "yuck", "ew", "eww",
    "creep", "creepy", "rotten", "stink", "stinks", "stinky",
}

EMOJI = {
    "joy": {"😀", "😄", "😁", "😊", "😍", "🥳", "😌", "🙂", ":)", ":-)", ":D", ":-D", "<3"},
    "sadness": {"😢", "😭", "☹️", "🙁", "😞", "😔", ":(", ":-(", ":'(", "T_T"},
    "anger": {"😠", "😡", ">:(", "!!1"},
    "fear": {"😨", "😰", "😱", "😬"},
    "disgust": {"🤢", "🤮"},
}

NEGATIONS = {
    "not", "no", "never", "hardly", "barely", "without",
    "isnt", "isn't", "dont", "don't", "cant", "can't", "wont", "won't",
}
BOOSTERS = {"very", "really", "so", "extremely", "super", "incredibly", "totally", "absolutely", "quite"}
DAMPENERS = {"slightly", "somewhat", "kinda", "kind", "sort", "sorta", "a", "bit", "little"}
CONTRASTIVE = {"but", "however", "though", "although", "yet", "nevertheless", "nonetheless"}
FIRST_PERSON = {"i", "im", "i'm", "ive", "i've", "me", "my", "mine"}

TOKEN_RE = re.compile(r"[a-zA-Z']+|[^\w\s]", re.UNICODE)


def _normalize_elongation(text: str) -> str:
    # "soooo goooood!!!" -> "soo good!!"
    return re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)


def _tokens(text: str) -> List[str]:
    text = _normalize_elongation(text)
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _stem(tok: str) -> str:
    # Tiny stemmer for common suffixes
    if not tok.isalpha():
        return tok
    for suf in ("'s", "ing", "ed", "ly", "ness", "ful", "ment", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def _window(tokens: List[str], i: int, size: int = 3) -> Iterable[str]:
    start = max(0, i - size)
    return tokens[start:i]


def _emoji_boost(tok: str) -> Dict[str, float]:
    scores = {k: 0.0 for k in ("joy", "sadness", "anger", "fear", "disgust")}
    for emo, bag in EMOJI.items():
        if tok in bag:
            scores[emo] += 1.2
    return scores


def _punctuation_emphasis(ahead: str) -> float:
    # more exclamation suggests higher arousal
    bangs = ahead.count("!")
    if bangs >= 3:
        return 1.3
    if bangs == 2:
        return 1.2
    if bangs == 1:
        return 1.1
    return 1.0


def _sarcasm_cue(tokens: List[str]) -> bool:
    text = " ".join(tokens)
    cues = ["yeah right", "as if", "sure buddy", "sure jan"]
    return any(c in text for c in cues)


def _lex_hit(stem: str) -> Dict[str, float]:
    scores = {k: 0.0 for k in ("joy", "sadness", "anger", "fear", "disgust")}
    if stem in JOY:
        scores["joy"] += 1.0
    if stem in SADNESS:
        scores["sadness"] += 1.0
    if stem in ANGER:
        scores["anger"] += 1.0
    if stem in FEAR:
        scores["fear"] += 1.0
    if stem in DISGUST:
        scores["disgust"] += 1.0
    return scores


def _merge(a: Dict[str, float], b: Dict[str, float], k: float = 1.0) -> None:
    for key in a:
        a[key] += b.get(key, 0.0) * k


def _score_clause(tokens: List[str]) -> Dict[str, float]:
    """Score a single clause of tokens."""
    scores = {k: 0.0 for k in ("joy", "sadness", "anger", "fear", "disgust")}
    n_alpha = 0

    for i, raw in enumerate(tokens):
        stem = _stem(raw)
        if raw.isalpha():
            n_alpha += 1

        # Emoji and emoticons
        _merge(scores, _emoji_boost(raw))

        # Lexical hit with local modifiers
        base = _lex_hit(stem)
        if any(base.values()):
            weight = 1.0

            w = list(_window(tokens, i, size=3))
            if any(wt in BOOSTERS for wt in w):
                weight *= 1.35
            if any(wt in DAMPENERS for wt in w):
                weight *= 0.65
            if any(wt in NEGATIONS for wt in w):
                weight *= -0.9

            # Punctuation and capitals
            tail = "".join(tokens[i:i + 4])
            weight *= _punctuation_emphasis(tail)
            if raw.isalpha() and len(raw) >= 3 and raw.upper() == raw:
                weight *= 1.15

            # Apply
            for k, v in base.items():
                if v:
                    scores[k] += v * weight

    # Negation inversion was already applied locally. Convert residual negatives.
    for emo, opp in [("joy", "sadness"), ("fear", "anger"), ("anger", "fear"), ("disgust", "joy"), ("sadness", "joy")]:
        if scores[emo] < 0:
            scores[opp] += abs(scores[emo]) * 0.6
            scores[emo] = 0.0

    # Heuristic signals not tied to a single token
    if "?" in tokens:
        scores["fear"] += 0.2  # uncertainty often signals fear or worry
    if any(p in tokens for p in ("!", "!!")):
        scores["anger"] += 0.05  # mild arousal nudge

    # First person focus strengthens emotions slightly
    if any(t in FIRST_PERSON for t in tokens):
        for k in scores:
            scores[k] *= 1.05

    # Sarcasm lightly reduces positive signals
    if _sarcasm_cue(tokens):
        scores["joy"] *= 0.6

    # Normalize by number of alpha tokens
    denom = max(n_alpha, 1)
    for k in scores:
        scores[k] = scores[k] / denom

    return scores


def _split_clauses(tokens: List[str]) -> List[List[str]]:
    """Split tokens by contrastive markers so that later clauses receive more weight."""
    if not tokens:
        return [[]]

    clauses: List[List[str]] = []
    current: List[str] = []

    for t in tokens:
        if t in CONTRASTIVE:
            if current:
                clauses.append(current)
            current = []
        else:
            current.append(t)
    if current:
        clauses.append(current)
    return clauses or [tokens]


def _squash(x: float) -> float:
    """Map unbounded x to [0, 1] smoothly."""
    # logistic squash centered near 0 with slope tuned for small magnitudes
    return 1.0 / (1.0 + math.exp(-4.0 * x))


def _clamp_scores(raw: Dict[str, float]) -> Dict[str, float]:
    # Scale and clamp to [0, 1]
    out = {}
    for k, v in raw.items():
        out[k] = max(0.0, min(_squash(v), 1.0))
    return out


def _aggregate_clauses(clauses: List[List[str]]) -> Dict[str, float]:
    """Score each clause and combine with increasing weights for later clauses."""
    raw = {k: 0.0 for k in ("joy", "sadness", "anger", "fear", "disgust")}
    total_w = 0.0
    for idx, clause in enumerate(clauses):
        sc = _score_clause(clause)
        # Weight later clauses more to capture "but now I feel X"
        weight = 1.0 + idx * 0.25
        total_w += weight
        for k in raw:
            raw[k] += sc[k] * weight

    if total_w == 0:
        return {k: 0.0 for k in raw}
    for k in raw:
        raw[k] /= total_w
    return raw


def _choose_dominant(scores: Dict[str, float]) -> str:
    """Pick a dominant emotion, or N/A if evidence is weak or tied."""
    vals = list(scores.values())
    if sum(vals) < 0.01 or max(vals) < 0.08:
        return "N/A"
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ordered) >= 2 and ordered[0][1] - ordered[1][1] < 0.04:
        return "N/A"
    return ordered[0][0]


def _fallback_model(text: str) -> Dict[str, float]:
    tokens = _tokens(text)
    if not tokens:
        return {k: 0.0 for k in ("anger", "disgust", "fear", "joy", "sadness")}

    clauses = _split_clauses(tokens)
    raw = _aggregate_clauses(clauses)
    return {
        "anger": float(_clamp_scores({"anger": raw["anger"]})["anger"]),
        "disgust": float(_clamp_scores({"disgust": raw["disgust"]})["disgust"]),
        "fear": float(_clamp_scores({"fear": raw["fear"]})["fear"]),
        "joy": float(_clamp_scores({"joy": raw["joy"]})["joy"]),
        "sadness": float(_clamp_scores({"sadness": raw["sadness"]})["sadness"]),
    }


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

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
