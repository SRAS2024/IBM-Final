# advanced_detector.py
# High fidelity local emotion detector with seven core dimensions and rich nuance.
# Public API:
#   detect_emotions(text: str) -> EmotionResult
#   explain_emotions(text: str) -> Dict[str, Any]  # debug-friendly trace
#
# Design goals
# 1) Keep the seven core scores (anger, disgust, fear, joy, sadness, passion, surprise).
# 2) Robust multi-sentence analysis up to twelve sentences, then clauses.
# 3) Specialized detectors for desire and commitment phrases.
# 4) Broaden lexical coverage with stems, phrases, idioms, emoji, and punctuation.
# 5) Pure standard library heuristics. Deterministic and side effect free.

from __future__ import annotations

import math
import os
import re
import difflib
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Tuple, Optional
from functools import lru_cache

# Optional Watson imports. If present and credentials are set,
# we can augment five classical emotions with our two computed ones.
try:  # pragma: no cover
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
except Exception:  # pragma: no cover
    NaturalLanguageUnderstandingV1 = None  # type: ignore
    IAMAuthenticator = None  # type: ignore
    Features = None  # type: ignore
    EmotionOptions = None  # type: ignore


# =============================================================================
# Data model
# =============================================================================

CORE_KEYS = ("anger", "disgust", "fear", "joy", "sadness", "passion", "surprise")

@dataclass
class EmotionResult:
    anger: float
    disgust: float
    fear: float
    joy: float
    sadness: float
    passion: float
    surprise: float
    dominant_emotion: str

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# =============================================================================
# Credentials and external pathway
# =============================================================================

def _has_watson_credentials() -> bool:
    return bool(os.getenv("WATSON_NLU_APIKEY")) and bool(os.getenv("WATSON_NLU_URL"))


def _call_watson(text: str) -> Dict[str, float]:
    """Return five classical emotions from IBM Watson NLU if available."""
    if NaturalLanguageUnderstandingV1 is None:
        raise RuntimeError("Watson SDK not available")

    apikey = os.getenv("WATSON_NLU_APIKEY")
    url = os.getenv("WATSON_NLU_URL")
    if not apikey or not url:
        raise RuntimeError("Watson credentials missing")

    authenticator = IAMAuthenticator(apikey)
    nlu = NaturalLanguageUnderstandingV1(version="2022-04-07", authenticator=authenticator)
    nlu.set_service_url(url)
    result = nlu.analyze(
        text=text,
        features=Features(emotion=EmotionOptions(document=True))
    ).get_result()
    e = result["emotion"]["document"]["emotion"]
    return {
        "anger": float(e.get("anger", 0.0)),
        "disgust": float(e.get("disgust", 0.0)),
        "fear": float(e.get("fear", 0.0)),
        "joy": float(e.get("joy", 0.0)),
        "sadness": float(e.get("sadness", 0.0)),
    }


# =============================================================================
# Lexicons and resources
# =============================================================================

# Core lexicons with stems. Keep broad for coverage.
JOY = {
    "love", "loved", "loving", "lovely", "like", "liked", "likes",
    "happy", "happi", "joy", "joyful", "cheer", "cheerful", "glad",
    "content", "contented", "relief", "relieved", "calm", "serene",
    "peace", "peaceful", "hope", "hopeful", "optimism", "optimistic",
    "delight", "delighted", "delightful", "pleased", "smile", "laugh",
    "awesome", "amazing", "wonderful", "fantastic", "great", "good",
    "proud", "success", "win", "won", "best", "perfect",
    "beautiful", "brilliant", "cute", "enjoy", "enjoyed", "enjoying",
    "thrilled", "ecstatic", "elated", "euphoric", "uplift", "brighten",
    "grace", "gratitude", "grateful", "thankful", "support", "supported",
    "comfort", "comfortable", "comfy", "yay", "hurray", "woohoo",
}

SADNESS = {
    "sad", "sadden", "saddened", "down", "blue", "depress", "depressed",
    "depressing", "cry", "cried", "crying", "tear", "tearful", "teary",
    "lonely", "alone", "miserable", "heartbroken", "broken",
    "sorrow", "grief", "mourning", "bereaved", "remorse", "regret",
    "regretful", "sorry", "homesick", "melancholy", "gloomy",
    "hopeless", "helpless", "tired", "drained", "exhausted", "empty",
    "aching", "pain", "hurt", "forlorn", "downcast", "somber", "sombre",
    "ashamed", "shame", "loss", "losing", "miss", "missing",
}

ANGER = {
    "angry", "anger", "mad", "furious", "livid", "rage", "raging",
    "irritated", "annoyed", "annoying", "upset", "hate", "hated", "hates",
    "hating", "outraged", "resentful", "hostile", "fume", "yell", "shout",
    "scream", "screaming", "frustrated", "frustrating", "infuriating",
    "disrespect", "insult", "insulted", "offended", "betrayed",
    "backstabbed", "lied", "lying", "cheated", "deceived", "seethe",
    "seething", "boil", "boiling", "spite", "vengeful", "irate", "ticked",
    "tickedoff", "ragequit",
}

FEAR = {
    "scare", "scared", "afraid", "fear", "fearful", "terrified",
    "terrify", "anxious", "anxiety", "worry", "worried", "worrying",
    "panic", "panicked", "nervous", "phobia", "frighten", "frightened",
    "tense", "uneasy", "alarmed", "concerned", "concern", "dread",
    "spook", "shaky", "shaking", "uncertain", "doubt", "doubted",
    "doubting", "threat", "threatened", "unsafe", "unease",
    "apprehensive", "jitters", "restless", "paranoid",
}

DISGUST = {
    "disgust", "disgusted", "gross", "nasty", "revolting", "repulsed",
    "repulsive", "sicken", "sickened", "vile", "filthy", "dirty", "yuck",
    "ew", "eww", "creep", "creepy", "rotten", "stink", "stinks", "stinky",
    "abhorrent", "appalling", "offensive", "foul", "toxic",
    "contaminated", "putrid", "icky",
}

# Passion for romantic desire, devotion, attachment.
PASSION = {
    "passion", "passionate", "desire", "desiring", "yearn", "yearning",
    "longing", "craving", "infatuated", "infatuation", "obsessed",
    "devoted", "devotion", "adore", "adoring", "adored", "cherish",
    "cherished", "romance", "romantic", "smitten", "inlove", "soulmate",
    "crush", "flushed", "butterflies", "attracted", "allured",
    "enamored", "enamoured", "fond", "fondness", "chemistry", "spark",
    "magnetic", "commit", "committed", "commitment", "marry", "marriage",
    "engage", "engaged", "proposal", "propose", "fiance", "fiancé",
    "fiancee", "fiancée", "wed", "wedding", "husband", "wife", "partner",
    "girlfriend", "boyfriend",
}

# Surprise for novelty or sudden change.
SURPRISE = {
    "surprise", "surprised", "surprising", "astonish", "astonished",
    "astonishing", "amaze", "amazed", "amazing", "shocked", "shock",
    "unexpected", "suddenly", "whoa", "wow", "omg", "wtf", "gasp",
    "unbelievable", "no way", "what", "holy", "plot twist",
    "didnt expect", "didn't expect",
}

# Special intent verbs and constructions that imply commitment, desire,
# reassurance, or reduction of negative affect.
INTENT_COMMIT = {
    "want to marry", "want to propose", "plan to marry", "planning to marry",
    "intend to marry", "intend to propose", "i will marry", "i will propose",
    "ask her to marry", "ask him to marry", "ready to commit", "ready for marriage",
    "ready to settle", "settle down", "build a life", "start a family",
}
INTENT_DESIRE = {
    "i want", "i wanna", "i would love", "i like", "i love", "i adore",
    "i need", "cant wait", "can't wait", "dying to", "itching to",
    "i'm in love", "im in love", "i am in love", "in love with",
    "falling in love", "fell in love",
}
INTENT_REASSURE = {
    "it will be ok", "it will be okay", "we will be ok", "we will be okay",
    "everything will be fine", "i believe in us", "we will make it",
}
INTENT_DEESCALATE = {
    "let us calm down", "let's calm down", "take a breath", "breathe",
    "we can talk", "i am listening", "i am here", "no need to fight",
}

# Idioms and phrases mapped to emotions with weights.
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
    ("head over heels", "passion", 2.0),
    ("butterflies in my stomach", "passion", 1.6),
    ("madly in love", "passion", 2.0),
    ("in love", "passion", 2.2),
    ("i'm in love", "passion", 2.4),
    ("im in love", "passion", 2.4),
    ("i am in love", "passion", 2.4),
    ("in love with", "passion", 2.3),
    ("falling in love", "passion", 2.1),
    ("fell in love", "passion", 2.1),
    ("i could not believe", "surprise", 1.7),
    ("i can't believe", "surprise", 1.7),
    ("could not believe", "surprise", 1.7),
    ("i want to marry", "passion", 2.0),
    ("i want to marry you", "passion", 2.2),
    ("i want to marry her", "passion", 2.2),
    ("i want to marry him", "passion", 2.2),
]

# Emoji and emoticons mapped to emotions.
EMOJI = {
    "joy": {"😀", "😄", "😁", "😊", "🥳", "😌", "🙂", ":)", ":-)", ":D", ":-D"},
    "sadness": {"😢", "😭", "☹️", "🙁", "😞", "😔", ":(", ":-(", ":'(", "T_T"},
    "anger": {"😠", "😡", ">:(", "!!1"},
    "fear": {"😨", "😰", "😱", "😬"},
    "disgust": {"🤢", "🤮"},
    "passion": {"😍", "😘", "🥰", "❤️", "💖", "💘", "<3"},
    "surprise": {"😲", "😳", "😮", "🤯", "😦", "😧", "😯", "😵"},
}

# Linguistic controls and cues.
NEGATIONS = {
    "not", "no", "never", "hardly", "barely", "without", "lack", "lacking",
    "isnt", "isn't", "dont", "don't", "cant", "can't", "wont", "won't", "aint", "ain't",
}
INTENSIFIERS = {
    "very", "really", "so", "extremely", "super", "incredibly", "totally",
    "absolutely", "quite", "truly", "deeply", "utterly", "highly", "too",
}
DAMPENERS = {"slightly", "somewhat", "kinda", "kind", "sort", "sorta", "a", "bit", "little", "mildly", "barely"}
HEDGES = {"maybe", "perhaps", "possibly", "i guess", "i suppose", "i think", "sort of", "kind of", "kinda", "ish"}
CONTRASTIVE = {"but", "however", "though", "although", "yet", "nevertheless", "nonetheless", "still", "even so"}
TEMPORAL_POS = {"now", "finally", "at last"}
TEMPORAL_NEG = {"still", "yet", "anymore", "no longer", "any longer"}
STANCE_1P = {"i", "im", "i'm", "ive", "i've", "me", "my", "mine", "we", "our", "ours"}

# Bigram negations that flip polarity strongly.
NEGATED_PAIRS = {
    ("no", "joy"): ("joy", "sadness", 1.1),
    ("no", "hope"): ("joy", "sadness", 1.1),
    ("without", "hope"): ("joy", "sadness", 1.0),
    ("not", "happy"): ("joy", "sadness", 1.0),
    ("not", "angry"): ("anger", "fear", 0.8),
    ("no", "love"): ("passion", "sadness", 1.0),
    ("not", "inlove"): ("passion", "sadness", 1.0),
    ("not", "in"): ("passion", "sadness", 0.9),  # handles "not in love"
}

# Regex for simple tokenization.
TOKEN_RE = re.compile(r"[a-zA-Z']+|[^\w\s]", re.UNICODE)

# Misspelling map and approximate targets.
MISSPELLINGS = {
    "hapy": "happy", "happpy": "happy", "happ": "happy",
    "angy": "angry", "angree": "angry",
    "discusting": "disgusting", "discust": "disgust",
    "woried": "worried", "anxios": "anxious", "scarry": "scary",
    "lonley": "lonely", "miserible": "miserable",
    "wierd": "weird", "definately": "definitely", "releived": "relieved",
    "beleive": "believe", "cant": "can't", "wont": "won't", "im": "i'm",
}
APPROX_TARGETS = set().union(JOY, SADNESS, ANGER, FEAR, DISGUST, PASSION, SURPRISE)


# =============================================================================
# Tokenization helpers
# =============================================================================

def _normalize_elongation(text: str) -> str:
    # sooooo goooood!!! -> soo good!!
    return re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> List[str]:
    text = _normalize_elongation(text)
    text = _normalize_whitespace(text)
    # collapse doubled intensifiers
    text = re.sub(r"\b(so|very|really)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
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
    if stem in APPROX_TARGETS:
        return stem
    if stem in MISSPELLINGS:
        return MISSPELLINGS[stem]
    matches = difflib.get_close_matches(stem, APPROX_TARGETS, n=1, cutoff=0.90)
    return matches[0] if matches else stem


# =============================================================================
# Sentence and clause splitting
# =============================================================================

_SENT_ENDERS = {".", "!", "?", "?!", "!?","\n"}

def _split_sentences_from_tokens(tokens: List[str], max_sentences: int = 12) -> List[List[str]]:
    """Split tokens into sentences. Cap at 12; merge overflow into the last sentence."""
    if not tokens:
        return [[]]
    sents: List[List[str]] = []
    current: List[str] = []
    for i, t in enumerate(tokens):
        current.append(t)
        if t in _SENT_ENDERS or (t == ";" and len(current) >= 6):
            sents.append(current)
            current = []
    if current:
        sents.append(current)
    if not sents:
        sents = [tokens[:]]

    if len(sents) <= max_sentences:
        return sents

    # Merge any overflow into the twelfth sentence to preserve information
    keep = sents[:max_sentences]
    overflow: List[str] = []
    for s in sents[max_sentences:]:
        overflow.extend(s)
    keep[-1].extend(overflow)
    return keep


def _split_clauses_in_sentence(sent_tokens: List[str]) -> List[List[str]]:
    """Split a sentence into clauses on contrastive markers and strong commas."""
    if not sent_tokens:
        return [[]]
    clauses: List[List[str]] = []
    current: List[str] = []
    for i, t in enumerate(sent_tokens):
        # Clause break on explicit discourse markers
        if t in CONTRASTIVE:
            if current:
                clauses.append(current)
            current = []
            continue
        # Clause break on comma/semicolon if current is long enough to stand alone
        if t in {",", ";", ":"} and len(current) >= 6:
            current.append(t)
            clauses.append(current)
            current = []
            continue
        current.append(t)

    if current:
        clauses.append(current)

    return clauses or [sent_tokens]


# =============================================================================
# Scoring utilities
# =============================================================================

def _blank_scores() -> Dict[str, float]:
    return {k: 0.0 for k in CORE_KEYS}


def _merge(acc: Dict[str, float], inc: Dict[str, float], scale: float = 1.0) -> None:
    for k in CORE_KEYS:
        acc[k] += inc.get(k, 0.0) * scale


def _emoji_boost(tok: str) -> Dict[str, float]:
    scores = _blank_scores()
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


def _surprise_punctuation_bonus(text: str) -> float:
    if re.search(r"(\?\!|\!\?)", text):
        return 0.15
    if re.search(r"[?!]{2,}", text):
        return 0.08
    return 0.0


def _in_lex(target: str, bag: set[str]) -> bool:
    if target in bag:
        return True
    if len(target) >= 4:
        return any(w.startswith(target) for w in bag)
    return False


def _lex_hit(stem: str) -> Dict[str, float]:
    s = _blank_scores()
    if _in_lex(stem, JOY): s["joy"] += 1.0
    if _in_lex(stem, SADNESS): s["sadness"] += 1.0
    if _in_lex(stem, ANGER): s["anger"] += 1.0
    if _in_lex(stem, FEAR): s["fear"] += 1.0
    if _in_lex(stem, DISGUST): s["disgust"] += 1.0
    if _in_lex(stem, PASSION): s["passion"] += 1.0
    if _in_lex(stem, SURPRISE): s["surprise"] += 1.0
    return s


def _apply_phrases(text_lower: str) -> Dict[str, float]:
    out = _blank_scores()
    for phrase, emo, w in PHRASES:
        if phrase in text_lower:
            out[emo] += w
    return out


def _apply_negated_pairs(tokens: List[str], scores: Dict[str, float]) -> None:
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        if bigram in NEGATED_PAIRS:
            src, dst, mult = NEGATED_PAIRS[bigram]
            scores[dst] += 0.9 * mult
            scores[src] *= 0.6


def _rhetorical_question_boost(tokens: List[str]) -> float:
    text = " ".join(tokens)
    cues = [
        "what if", "what happens if", "am i going to", "are we going to",
        "is it going to", "how will i", "how can i", "why does this always",
    ]
    return 0.25 if any(c in text for c in cues) else 0.0


def _because_clause_dampener(tokens: List[str]) -> float:
    text = " ".join(tokens)
    if any(k in text for k in ("because", "since", "as ", "due to")):
        return 0.9
    return 1.0


def _harvest_meta_counts(tokens: List[str]) -> Dict[str, float]:
    joined = "".join(tokens)
    return {
        "_exclam_count": float(joined.count("!")),
        "_dots_count": float(joined.count("...")),
        "_heart_count": float(joined.count("❤️") + joined.count("<3")),
        "_laugh_count": float(joined.count("haha") + joined.count("lol")),
    }


def _arousal_valence_nudge(scores: Dict[str, float]) -> None:
    anger = scores.get("anger", 0.0)
    fear = scores.get("fear", 0.0)
    joy = scores.get("joy", 0.0)
    passion = scores.get("passion", 0.0)
    sadness = scores.get("sadness", 0.0)
    disgust = scores.get("disgust", 0.0)

    exclam = scores.pop("_exclam_count", 0.0)
    dots = scores.pop("_dots_count", 0.0)
    hearts = scores.pop("_heart_count", 0.0)
    laughs = scores.pop("_laugh_count", 0.0)

    high_arousal = min(exclam * 0.01, 0.05)
    low_arousal = min(dots * 0.01, 0.05)
    warm = min((hearts + laughs) * 0.01, 0.05)

    scores["anger"] = anger * (1.0 + high_arousal)
    scores["fear"] = fear * (1.0 + high_arousal)
    scores["joy"] = joy * (1.0 + warm)
    scores["passion"] = passion * (1.0 + warm)
    scores["sadness"] = sadness * (1.0 + low_arousal)
    scores["disgust"] = disgust * (1.0 + low_arousal)


# =============================================================================
# Clause and sentence scorers
# =============================================================================

def _desire_commitment_bonus(text_lower: str) -> Dict[str, float]:
    """Detect desire or commitment phrases and boost passion and joy."""
    out = _blank_scores()
    if any(p in text_lower for p in INTENT_COMMIT):
        out["passion"] += 2.3
        out["joy"] += 0.6
        out["anger"] -= 0.4
    if any(p in text_lower for p in INTENT_DESIRE):
        out["passion"] += 1.6
        out["joy"] += 0.4
    if " in love" in text_lower or text_lower.startswith("in love"):
        out["passion"] += 2.2
        out["joy"] += 0.4
    if any(p in text_lower for p in INTENT_REASSURE):
        out["joy"] += 0.7
        out["fear"] *= 0.8
    if any(p in text_lower for p in INTENT_DEESCALATE):
        out["anger"] *= 0.7
        out["fear"] *= 0.9
        out["joy"] += 0.2
    return out


def _scope_has_negation(win_before: Iterable[str]) -> bool:
    return any(wt in NEGATIONS or wt.endswith("n't") for wt in win_before)


def _score_clause(tokens: List[str]) -> Dict[str, float]:
    """Score a single clause with lexical, emoji, punctuation, and intent cues."""
    scores = _blank_scores()
    n_alpha = 0

    text_lower = " ".join(tokens)
    _merge(scores, _apply_phrases(text_lower))
    _merge(scores, _desire_commitment_bonus(text_lower), 1.0)

    scores.update(_harvest_meta_counts(tokens))

    for i, raw in enumerate(tokens):
        stem = _stem(raw)
        if raw.isalpha():
            n_alpha += 1

        # Direct emoji
        _merge(scores, _emoji_boost(raw))

        stem = _approx_correction(stem)
        base = _lex_hit(stem)
        if any(base.values()):
            weight = 1.0

            # Context windows for local modifiers
            win = list(_window(tokens, i, size=3))

            # Intensifiers, dampeners, hedges
            if any(wt in INTENSIFIERS for wt in win): weight *= 1.35
            if any(wt in DAMPENERS for wt in win): weight *= 0.65
            if any(wt in HEDGES for wt in win): weight *= 0.88

            # Negation scope
            if _scope_has_negation(win):
                weight *= -0.9

            # Temporal and first person stance
            if any(wt in TEMPORAL_POS for wt in win): weight *= 1.05
            if any(wt in TEMPORAL_NEG for wt in win): weight *= 0.95
            if any(wt in STANCE_1P for wt in win): weight *= 1.05

            # Punctuation emphasis and shouting
            tail = "".join(tokens[i:i + 4])
            weight *= _punctuation_emphasis(tail)
            if raw.isalpha() and len(raw) >= 3 and raw.upper() == raw:
                weight *= 1.1

            # Apply base with computed weight
            for k, v in base.items():
                if v:
                    scores[k] += v * weight

    # Global clause cues
    rq_boost = _rhetorical_question_boost(tokens)
    if rq_boost:
        scores["fear"] += rq_boost

    s_damp = _because_clause_dampener(tokens)
    scores["surprise"] *= s_damp

    # Redirect negative residues to opposites, then zero them
    for emo, opp in [
        ("joy", "sadness"),
        ("fear", "anger"),
        ("anger", "fear"),
        ("disgust", "joy"),
        ("sadness", "joy"),
        ("passion", "sadness"),
        ("surprise", "fear"),
    ]:
        if scores[emo] < 0:
            scores[opp] += abs(scores[emo]) * 0.6
            scores[emo] = 0.0

    # Mild punctuation nudges
    if "?" in tokens:
        scores["fear"] += 0.2
        scores["joy"] *= 0.98
    if any(p in tokens for p in ("!", "!!")):
        scores["anger"] += 0.03

    # First person increases salience slightly
    if any(t in STANCE_1P for t in tokens):
        for k in scores:
            scores[k] *= 1.03

    # Sarcasm dampens joy
    if _sarcasm_cue(tokens):
        scores["joy"] *= 0.6

    # Shouting cue across clause
    text_seg = "".join(tokens)
    if re.search(r"\b[A-Z]{4,}\b", text_seg):
        scores["anger"] *= 1.07
        scores["joy"] *= 1.05

    # Length normalization
    denom = max(n_alpha, 1)
    for k in CORE_KEYS:
        scores[k] = scores[k] / denom

    # Structured negated pairs like "not happy"
    _apply_negated_pairs(tokens, scores)

    # Surprise punctuation bonus
    if _surprise_punctuation_bonus("".join(tokens)):
        scores["surprise"] += 0.05

    # Arousal and valence nudges from meta counts
    _arousal_valence_nudge(scores)

    return scores


def _sarcasm_cue(tokens: List[str]) -> bool:
    text = " ".join(tokens)
    cues = [
        "yeah right", "as if", "sure buddy", "sure jan", "what a joy",
        "great job", "so fun", "how lovely", "what a delight",
    ]
    return any(c in text for c in cues)


def _sentence_emphasis_weight(tokens: List[str], idx: int, n_sent: int) -> float:
    """Compute a soft emphasis weight for a sentence."""
    text = "".join(tokens)
    bangs = text.count("!")
    qmarks = text.count("?")
    caps_tokens = sum(1 for t in tokens if t.isalpha() and len(t) >= 3 and t.upper() == t)
    alpha_tokens = sum(1 for t in tokens if t.isalpha())
    caps_ratio = (caps_tokens / alpha_tokens) if alpha_tokens else 0.0

    punct_boost = 1.0 + min(bangs * 0.03 + qmarks * 0.01, 0.12)
    caps_boost = 1.0 + min(caps_ratio * 0.20, 0.10)
    pos_boost = 1.0 + min((idx / max(1, n_sent - 1)) * 0.10, 0.10)  # slightly reward later sentences

    return punct_boost * caps_boost * pos_boost


# =============================================================================
# Aggregation and post processing
# =============================================================================

def _squash(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-4.0 * x))


def _clamp_scores(raw: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in raw.items():
        out[k] = max(0.0, min(_squash(v), 1.0))
    return out


def _aggregate_sentence(sent_tokens: List[str]) -> Dict[str, float]:
    """Aggregate clause scores within a sentence with clause-level emphasis."""
    clause_scores: List[Tuple[Dict[str, float], float]] = []
    clauses = _split_clauses_in_sentence(sent_tokens)
    for cl in clauses:
        sc = _score_clause(cl)
        # Clause emphasis: punctuation and length give a soft emphasis
        tail = "".join(cl[-4:]) if cl else ""
        emph = _punctuation_emphasis(tail)
        # Very short clauses get slightly damped to avoid single-phrase bias
        alpha_len = sum(1 for t in cl if t.isalpha())
        len_boost = 0.9 if alpha_len <= 3 else 1.0
        clause_scores.append((sc, emph * len_boost))

    if not clause_scores:
        return _blank_scores()

    # Weighted average across clauses
    out = _blank_scores()
    total_w = sum(w for _, w in clause_scores) or 1.0
    for sc, w in clause_scores:
        for k in CORE_KEYS:
            out[k] += sc[k] * (w / total_w)
    return out


def _aggregate_sentences(sentences: List[List[str]]) -> Dict[str, float]:
    """Aggregate sentence scores with emphasis and anti dominance smoothing."""
    if not sentences:
        return _blank_scores()

    n = len(sentences)
    per_sent: List[Tuple[Dict[str, float], float]] = []
    for idx, s in enumerate(sentences):
        sc = _aggregate_sentence(s)
        w = _sentence_emphasis_weight(s, idx, n)
        per_sent.append((sc, w))

    # Normalize weights and apply anti dominance smoothing for many sentences
    weights = [w for _, w in per_sent]
    sum_w = sum(weights) or 1.0
    weights = [w / sum_w for w in weights]

    if n >= 5:
        # Blend with uniform weights to prevent any one sentence from dominating
        uni = 1.0 / n
        weights = [0.6 * w + 0.4 * uni for w in weights]

    # Recompute normalized weights
    sum_w = sum(weights) or 1.0
    weights = [w / sum_w for w in weights]

    raw = _blank_scores()
    for (sc, _), w in zip(per_sent, weights):
        for k in CORE_KEYS:
            raw[k] += sc[k] * w
    return raw


def _choose_dominant(scores: Dict[str, float]) -> str:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_val = ordered[0][1]
    second_val = ordered[1][1]
    if top_val < 0.10:
        return "N/A"
    if top_val - second_val < 0.02:
        return ordered[0][0]
    return ordered[0][0]


def _augment_watson_with_two(text: str, five: Dict[str, float]) -> Dict[str, float]:
    tokens = _tokens(text)
    sentences = _split_sentences_from_tokens(tokens, max_sentences=12)
    raw = _aggregate_sentences(sentences)
    passion = max(0.0, min(raw.get("passion", 0.0), 1.0))
    surprise = max(0.0, min(raw.get("surprise", 0.0), 1.0))
    out = dict(five)
    out["passion"] = float(passion)
    out["surprise"] = float(surprise)
    return out


# =============================================================================
# Public API
# =============================================================================

def detect_emotions(text: str, use_watson_if_available: bool = True) -> EmotionResult:
    """Detect emotions and return seven scores plus dominant core."""
    if text is None or not str(text).strip():
        raise ValueError("Input text is required")

    if use_watson_if_available and _has_watson_credentials():
        five = _call_watson(text)
        scores = _augment_watson_with_two(text, five)
    else:
        tokens = _tokens(text)
        if not tokens:
            scores = _blank_scores()
        else:
            sentences = _split_sentences_from_tokens(tokens, max_sentences=12)
            raw = _aggregate_sentences(sentences)
            # Global surprise punctuation bonus across the whole text
            raw["surprise"] += _surprise_punctuation_bonus("".join(tokens)) * 0.2
            scores = _clamp_scores(raw)

    result = EmotionResult(
        anger=float(scores["anger"]),
        disgust=float(scores["disgust"]),
        fear=float(scores["fear"]),
        joy=float(scores["joy"]),
        sadness=float(scores["sadness"]),
        passion=float(scores["passion"]),
        surprise=float(scores["surprise"]),
        dominant_emotion=_choose_dominant(scores),
    )
    return result


def explain_emotions(text: str, use_watson_if_available: bool = False) -> Dict[str, Any]:
    """Return a debug dictionary with intermediate artifacts for transparency."""
    tokens = _tokens(text)
    sentences = _split_sentences_from_tokens(tokens, max_sentences=12)
    per_sentence = []
    per_sentence_weights = []
    for idx, s in enumerate(sentences):
        sc_sentence = _aggregate_sentence(s)
        per_sentence.append({
            "sentence_tokens": s,
            "sentence_scores": sc_sentence,
            "clauses": _split_clauses_in_sentence(s),
        })
        per_sentence_weights.append(_sentence_emphasis_weight(s, idx, len(sentences)))

    # Normalize weights like detector path
    sw = per_sentence_weights[:]
    total = sum(sw) or 1.0
    sw = [w / total for w in sw]
    if len(sw) >= 5:
        uni = 1.0 / len(sw)
        sw = [0.6 * w + 0.4 * uni for w in sw]
    total = sum(sw) or 1.0
    sw = [w / total for w in sw]

    agg = _aggregate_sentences(sentences) if sentences else _blank_scores()
    agg["surprise"] += _surprise_punctuation_bonus("".join(tokens)) * 0.2
    final = _clamp_scores(agg)
    return {
        "text": text,
        "tokens": tokens,
        "sentences": sentences,
        "per_sentence": per_sentence,
        "sentence_weights": sw,
        "aggregate_scores": agg,
        "final_scores": final,
        "dominant": _choose_dominant(final),
    }

# Back-compat alias if anything imports `emotion_detector` directly from here.
emotion_detector = detect_emotions
