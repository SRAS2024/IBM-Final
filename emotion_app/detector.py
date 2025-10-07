"""High fidelity local emotion detector with robust offline heuristics.

Goals
1) Fully offline. No external services required.
2) Handle short phrases, partial sentences, typos, emoji, and emphasis.
3) Model common linguistic phenomena:
   - negation scope
   - intensifiers and dampeners
   - contrastive pivots such as "but", "however"
   - temporal cues such as "now", "still", "no longer"
   - emphasis from capitalization, repetition, punctuation
   - sarcasm and hedging cues
   - first person stance
   - phrase idioms and multiword expressions
4) Produce stable scores in [0, 1] and a sensible dominant emotion.
5) Keep the public API unchanged for the rest of the app.

If IBM Watson NLU credentials are present, Watson is used automatically.
Otherwise this file provides an advanced rule based fallback.
"""

from __future__ import annotations

import math
import os
import re
import difflib
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

from .errors import InvalidTextError, ServiceUnavailableError

# Optional Watson imports for environments that have credentials.
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

# Core lexicons. Lowercased stems. Expanded for higher coverage.
JOY = {
    "love", "loved", "loving", "like", "liked", "likes",
    "happy", "happi", "joy", "joyful", "cheer", "cheerful",
    "content", "contented", "excite", "excited", "exciting",
    "glad", "delight", "delighted", "delightful", "pleased",
    "satisfy", "satisfied", "awesome", "amazing", "wonderful",
    "fantastic", "great", "good", "grateful", "thankful",
    "blessed", "peaceful", "calm", "serene", "hopeful",
    "win", "won", "success", "proud", "smile", "laugh",
    "yay", "woohoo", "hurray", "enjoy", "enjoyed", "enjoying",
    "fun", "cute", "beautiful", "brilliant", "best", "perfect",
    "encourage", "optimistic", "optimism", "relief", "relieved",
    "comfort", "comfortable", "comfy", "support", "supported",
    "thrilled", "ecstatic", "elated", "glowing", "uplift", "brighten",
}
SADNESS = {
    "sad", "sadden", "saddened", "down", "depress", "depressed", "depressing",
    "cry", "cried", "crying", "tear", "tearful", "teary", "lonely", "alone",
    "miserable", "heartbroken", "sorrow", "grief", "blue", "hopeless",
    "helpless", "gloomy", "lost", "regret", "regretful", "sorry", "miss",
    "missing", "tired", "drained", "exhausted", "empty", "broken",
    "devastated", "melancholy", "homesick", "aching", "pain", "hurt",
    "downcast", "forlorn", "sombre", "somber",
}
ANGER = {
    "angry", "anger", "mad", "furious", "rage", "raging", "irritated", "annoyed",
    "annoying", "upset", "hate", "hated", "hates", "hating", "outraged", "livid",
    "resentful", "hostile", "fume", "yell", "shout", "scream", "screaming",
    "frustrated", "frustrating", "infuriating", "disrespect", "insult", "insulted",
    "offended", "betrayed", "backstabbed", "lied", "lying", "cheated", "deceived",
    "ragequit", "rageful", "enraged", "fed", "fedup", "ticked", "tickedoff",
    "irate", "seethe", "seething", "boil", "boiling",
}
FEAR = {
    "scare", "scared", "afraid", "fear", "fearful", "terrified", "terrify",
    "anxious", "anxiety", "worry", "worried", "worrying", "panic", "panicked",
    "nervous", "phobia", "frighten", "frightened", "tense", "uneasy", "alarmed",
    "concerned", "concern", "dread", "spook", "shaky", "shaking", "uncertain",
    "doubt", "doubted", "doubting", "threat", "threatened", "unsafe",
    "unease", "apprehensive", "jitters", "restless",
}
DISGUST = {
    "disgust", "disgusted", "gross", "nasty", "revolting", "repulsed", "repulsive",
    "sicken", "sickened", "vile", "filthy", "dirty", "yuck", "ew", "eww",
    "creep", "creepy", "rotten", "stink", "stinks", "stinky", "abhorrent",
    "appalling", "offensive", "foul", "toxic", "contaminated", "putrid",
}

# Multiword expressions and idioms mapped to emotions with weights.
PHRASES: List[Tuple[str, str, float]] = [
    ("over the moon", "joy", 1.8),
    ("on cloud nine", "joy", 1.6),
    ("could not be happier", "joy", 1.9),
    ("couldn't be happier", "joy", 1.9),
    ("walking on air", "joy", 1.6),
    ("in tears", "sadness", 1.4),
    ("cry my eyes out", "sadness", 1.9),
    ("heart is broken", "sadness", 1.9),
    ("i feel empty", "sadness", 1.6),
    ("boiling with rage", "anger", 2.0),
    ("lost my temper", "anger", 1.7),
    ("at my wits end", "anger", 1.5),
    ("out of my mind with worry", "fear", 1.9),
    ("sick to my stomach", "disgust", 1.7),
    ("gives me the creeps", "disgust", 1.7),
    ("creeps me out", "disgust", 1.7),
]

# Emoji and emoticons mapped to emotions.
EMOJI = {
    "joy": {"😀", "😄", "😁", "😊", "😍", "🥳", "😌", "🙂", ":)", ":-)", ":D", ":-D", "<3"},
    "sadness": {"😢", "😭", "☹️", "🙁", "😞", "😔", ":(", ":-(", ":'(", "T_T"},
    "anger": {"😠", "😡", ">:(", "!!1"},
    "fear": {"😨", "😰", "😱", "😬"},
    "disgust": {"🤢", "🤮"},
}

# Linguistic controls and cues
NEGATIONS = {
    "not", "no", "never", "hardly", "barely", "without", "lack", "lacking",
    "isnt", "isn't", "dont", "don't", "cant", "can't", "wont", "won't",
}
INTENSIFIERS = {"very", "really", "so", "extremely", "super", "incredibly", "totally", "absolutely", "quite", "truly"}
DAMPENERS = {"slightly", "somewhat", "kinda", "kind", "sort", "sorta", "a", "bit", "little", "mildly"}
HEDGES = {"maybe", "perhaps", "possibly", "i guess", "i suppose", "i think", "sort of", "kind of"}
CONTRASTIVE = {"but", "however", "though", "although", "yet", "nevertheless", "nonetheless", "still"}
TEMPORAL_POS = {"now", "finally", "at last"}
TEMPORAL_NEG = {"still", "yet", "anymore", "no longer", "any longer"}
STANCE_1P = {"i", "im", "i'm", "ive", "i've", "me", "my", "mine"}

# Negated noun phrases that should flip polarity strongly when seen as bigrams
NEGATED_PAIRS = {
    ("no", "joy"): ("joy", "sadness", 1.1),
    ("no", "hope"): ("joy", "sadness", 1.1),
    ("without", "hope"): ("joy", "sadness", 1.0),
    ("not", "happy"): ("joy", "sadness", 1.0),
    ("not", "angry"): ("anger", "fear", 0.8),
}

TOKEN_RE = re.compile(r"[a-zA-Z']+|[^\w\s]", re.UNICODE)

# Misspelling map and approximate matching target vocab
MISSPELLINGS = {
    "hapy": "happy", "happpy": "happy", "happ": "happy",
    "angy": "angry", "angree": "angry",
    "discusting": "disgusting", "discust": "disgust",
    "woried": "worried", "anxios": "anxious", "scarry": "scary",
    "lonley": "lonely", "miserible": "miserable",
    "wierd": "weird", "definately": "definitely", "releived": "relieved",
}
APPROX_TARGETS = set().union(JOY, SADNESS, ANGER, FEAR, DISGUST)


def _normalize_elongation(text: str) -> str:
    # "soooo goooood!!!" -> "soo good!!"
    return re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)


def _tokens(text: str) -> List[str]:
    text = _normalize_elongation(text)
    # collapse common doubled words like "so so" which should act like a light intensifier
    text = re.sub(r"\b(so|very)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _stem(tok: str) -> str:
    if not tok.isalpha():
        return tok
    for suf in ("'s", "ing", "ed", "ly", "ness", "ful", "ment", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def _window(tokens: List[str], i: int, size: int = 3) -> Iterable[str]:
    start = max(0, i - size)
    return tokens[start:i]


@lru_cache(maxsize=4096)
def _approx_correction(stem: str) -> str:
    """Correct obvious typos with a small dictionary and fuzzy match fallback."""
    if stem in APPROX_TARGETS:
        return stem
    if stem in MISSPELLINGS:
        return MISSPELLINGS[stem]
    matches = difflib.get_close_matches(stem, APPROX_TARGETS, n=1, cutoff=0.90)
    return matches[0] if matches else stem


def _emoji_boost(tok: str) -> Dict[str, float]:
    scores = {k: 0.0 for k in ("joy", "sadness", "anger", "fear", "disgust")}
    for emo, bag in EMOJI.items():
        if tok in bag:
            scores[emo] += 1.3
    return scores


def _punctuation_emphasis(ahead: str) -> float:
    bangs = ahead.count("!")
    if bangs >= 3:
        return 1.35
    if bangs == 2:
        return 1.2
    if bangs == 1:
        return 1.1
    return 1.0


def _sarcasm_cue(tokens: List[str]) -> bool:
    text = " ".join(tokens)
    cues = [
        "yeah right", "as if", "sure buddy", "sure jan", "what a joy",
        "great job", "so fun", "how lovely", "what a delight",
    ]
    return any(c in text for c in cues)


def _lex_hit(stem: str) -> Dict[str, float]:
    scores = {k: 0.0 for k in ("joy", "sadness", "anger", "fear", "disgust")}
    # allow partial stems of length 4+ to match, improves coverage on partial words
    def in_lex(target: str, bag: set[str]) -> bool:
        if target in bag:
            return True
        if len(target) >= 4:
            return any(w.startswith(target) for w in bag)
        return False

    if in_lex(stem, JOY):
        scores["joy"] += 1.0
    if in_lex(stem, SADNESS):
        scores["sadness"] += 1.0
    if in_lex(stem, ANGER):
        scores["anger"] += 1.0
    if in_lex(stem, FEAR):
        scores["fear"] += 1.0
    if in_lex(stem, DISGUST):
        scores["disgust"] += 1.0
    return scores


def _merge(acc: Dict[str, float], inc: Dict[str, float], k: float = 1.0) -> None:
    for key in acc:
        acc[key] += inc.get(key, 0.0) * k


def _apply_phrases(text_lower: str) -> Dict[str, float]:
    out = {k: 0.0 for k in ("joy", "sadness", "anger", "fear", "disgust")}
    for phrase, emo, w in PHRASES:
        if phrase in text_lower:
            out[emo] += w
    return out


def _apply_negated_pairs(tokens: List[str], scores: Dict[str, float]) -> None:
    """Boost opposite emotion when explicit negated noun phrases appear."""
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        if bigram in NEGATED_PAIRS:
            src, dst, mult = NEGATED_PAIRS[bigram]
            scores[dst] += 0.9 * mult
            # also damp the source to avoid ties
            scores[src] *= 0.6


def _score_clause(tokens: List[str]) -> Dict[str, float]:
    """Score a single clause of tokens."""
    scores = {k: 0.0 for k in ("joy", "sadness", "anger", "fear", "disgust")}
    n_alpha = 0

    # Phrase level detection
    _merge(scores, _apply_phrases(" ".join(tokens)))

    for i, raw in enumerate(tokens):
        stem = _stem(raw)
        if raw.isalpha():
            n_alpha += 1

        # Emoji and emoticons
        _merge(scores, _emoji_boost(raw))

        # Lexical hit with local modifiers, typo correction, and emphasis
        stem = _approx_correction(stem)
        base = _lex_hit(stem)
        if any(base.values()):
            weight = 1.0

            win = list(_window(tokens, i, size=3))
            if any(wt in INTENSIFIERS for wt in win):
                weight *= 1.35
            if any(wt in DAMPENERS for wt in win):
                weight *= 0.65
            if any(wt in HEDGES for wt in win):
                weight *= 0.85

            # Negation inversion
            if any(wt in NEGATIONS for wt in win):
                weight *= -0.9

            # Temporal and stance nudges
            if any(wt in TEMPORAL_POS for wt in win):
                weight *= 1.05
            if any(wt in TEMPORAL_NEG for wt in win):
                weight *= 0.95
            if any(wt in STANCE_1P for wt in win):
                weight *= 1.05

            # Punctuation and capitals emphasis
            tail = "".join(tokens[i:i + 4])
            weight *= _punctuation_emphasis(tail)
            if raw.isalpha() and len(raw) >= 3 and raw.upper() == raw:
                weight *= 1.15

            # Apply
            for k, v in base.items():
                if v:
                    scores[k] += v * weight

    # Residual negative contributions to opposite signals
    for emo, opp in [("joy", "sadness"), ("fear", "anger"), ("anger", "fear"), ("disgust", "joy"), ("sadness", "joy")]:
        if scores[emo] < 0:
            scores[opp] += abs(scores[emo]) * 0.6
            scores[emo] = 0.0

    # Heuristic signals not tied to one token
    if "?" in tokens:
        scores["fear"] += 0.2
        scores["joy"] *= 0.98
    if any(p in tokens for p in ("!", "!!")):
        scores["anger"] += 0.05

    # First person across the clause strengthens intensity slightly
    if any(t in STANCE_1P for t in tokens):
        for k in scores:
            scores[k] *= 1.05

    # Sarcasm reduces joy
    if _sarcasm_cue(tokens):
        scores["joy"] *= 0.6

    # Extra rule for all caps segments that signal strong arousal
    text_seg = "".join(tokens)
    if re.search(r"\b[A-Z]{4,}\b", text_seg):
        scores["anger"] *= 1.1
        scores["joy"] *= 1.07

    # Normalize by number of alpha tokens
    denom = max(n_alpha, 1)
    for k in scores:
        scores[k] = scores[k] / denom

    # Explicit negated bigrams after normalization
    _apply_negated_pairs(tokens, scores)

    return scores


def _split_clauses(tokens: List[str]) -> List[List[str]]:
    """Split tokens by contrastive markers so that later clauses get more weight."""
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
    """Map unbounded x to [0, 1] smoothly with a logistic curve."""
    return 1.0 / (1.0 + math.exp(-4.0 * x))


def _clamp_scores(raw: Dict[str, float]) -> Dict[str, float]:
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
        # Weight later clauses more to capture final sentiment shifts.
        weight = 1.0 + idx * 0.3
        total_w += weight
        for k in raw:
            raw[k] += sc[k] * weight

    if total_w == 0:
        return {k: 0.0 for k in raw}
    for k in raw:
        raw[k] /= total_w
    return raw


def _choose_dominant(scores: Dict[str, float]) -> str:
    """Pick a dominant emotion or N/A when evidence is weak or tied."""
    vals = list(scores.values())
    if sum(vals) < 0.01 or max(vals) < 0.08:
        return "N/A"
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    # slightly stricter tie margin to avoid spurious winners
    if len(ordered) >= 2 and ordered[0][1] - ordered[1][1] < 0.05:
        return "N/A"
    return ordered[0][0]


def _fallback_model(text: str) -> Dict[str, float]:
    tokens = _tokens(text)
    if not tokens:
        return {k: 0.0 for k in ("anger", "disgust", "fear", "joy", "sadness")}

    clauses = _split_clauses(tokens)
    raw = _aggregate_clauses(clauses)

    # Final clamp to [0, 1]
    clamped = _clamp_scores(raw)
    return {
        "anger": float(clamped["anger"]),
        "disgust": float(clamped["disgust"]),
        "fear": float(clamped["fear"]),
        "joy": float(clamped["joy"]),
        "sadness": float(clamped["sadness"]),
    }


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

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

    dominant = _choose_dominant(scores)
    return EmotionResult(
        anger=scores.get("anger", 0.0),
        disgust=scores.get("disgust", 0.0),
        fear=scores.get("fear", 0.0),
        joy=scores.get("joy", 0.0),
        sadness=scores.get("sadness", 0.0),
        dominant_emotion=dominant,
    )
