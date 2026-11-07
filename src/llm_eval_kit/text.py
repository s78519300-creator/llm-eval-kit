"""Small, dependency-free text utilities shared by the checkers.

Everything here is deliberately simple and deterministic: the goal is to get
signals that are cheap, explainable and stable across runs, not to replace a
proper NLP pipeline.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

STOPWORDS = frozenset(
    """
    a an the and or but if then so of to in on at by for with from as is are was
    were be been being it its it's this that these those there here which who whom
    whose what when where why how do does did has have had having can could should
    would may might will shall must i you he she we they them his her their our
    your my me us also than such into over under about after before between
    through during up down out off very just only more most other some any each
    both all yes per via within while because however
    """.split()
)

NEGATIONS = frozenset(
    {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "cannot", "without"}
)

LSQUO = "\N{LEFT SINGLE QUOTATION MARK}"
RSQUO = "\N{RIGHT SINGLE QUOTATION MARK}"

_WORD_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?|[a-z]+(?:'[a-z]+)?")
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")


def normalize(text: str) -> str:
    """Lower-case text and normalise curly quotes to plain apostrophes."""
    return text.lower().replace(RSQUO, "'").replace(LSQUO, "'")


def normalize_number(raw: str) -> str:
    """Canonicalise a numeric string so ``1,200`` == ``1200`` and ``3.50`` == ``3.5``."""
    value = raw.replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value or "0"


def tokenize(text: str) -> list[str]:
    """Split text into lower-case word and number tokens.

    Numbers are normalised with :func:`normalize_number` so they compare
    equal regardless of thousands separators.
    """
    tokens = []
    for tok in _WORD_RE.findall(normalize(text)):
        tokens.append(normalize_number(tok) if tok[0].isdigit() else tok)
    return tokens


def is_negation(token: str) -> bool:
    """Return True for negation cues such as ``not``, ``never`` or ``isn't``."""
    return token in NEGATIONS or token.endswith("n't")


def has_negation(text: str) -> bool:
    """Return True if the text contains at least one negation cue."""
    return any(is_negation(tok) for tok in tokenize(text))


def stem(token: str) -> str:
    """Very light suffix stripping so that plurals match their singular form."""
    if token[0].isdigit():
        return token
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def content_tokens(text: str, keep_numbers: bool = True) -> list[str]:
    """Return the meaningful tokens of ``text`` in order.

    Stopwords, negation cues and single characters are dropped and the rest
    are lightly stemmed. Negations are removed on purpose: polarity is handled
    separately by the contradiction checker, and leaving them in would let
    "is not available" look like strong support for "is available".
    """
    out = []
    for tok in tokenize(text):
        if tok in STOPWORDS or is_negation(tok):
            continue
        if tok[0].isdigit():
            if keep_numbers:
                out.append(tok)
            continue
        if len(tok) < 2:
            continue
        out.append(stem(tok))
    return out


def extract_numbers(text: str) -> set[str]:
    """Return the set of normalised numbers mentioned in ``text``."""
    return {normalize_number(m) for m in _NUMBER_RE.findall(text)}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (and list items) using punctuation and newlines."""
    pieces = _SENTENCE_RE.split(text.strip())
    sentences = []
    for piece in pieces:
        cleaned = _BULLET_RE.sub("", piece).strip()
        if cleaned:
            sentences.append(cleaned)
    return sentences


_CLAUSE_RE = re.compile(
    r";\s*|,?\s+\b(?:so|but|and|whereas|while|although|though|yet)\b\s+", re.IGNORECASE
)


def split_clauses(sentence: str) -> list[str]:
    """Split a sentence into clauses at semicolons and common coordinating conjunctions.

    Comparing clause by clause keeps a negation in one half of a compound
    sentence from being attributed to the other half.
    """
    return [c.strip(" ,") for c in _CLAUSE_RE.split(sentence) if c.strip(" ,")]


def ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    """Return the set of contiguous ``n``-grams in ``tokens``."""
    if n <= 0:
        raise ValueError("n must be positive")
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def recall(part: Iterable[str], whole: Iterable[str]) -> float:
    """Fraction of distinct items in ``part`` that also appear in ``whole``."""
    part_set = set(part)
    if not part_set:
        return 1.0
    return len(part_set & set(whole)) / len(part_set)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity of two token collections."""
    a_set, b_set = set(a), set(b)
    if not a_set and not b_set:
        return 1.0
    return len(a_set & b_set) / len(a_set | b_set)


def shorten(text: str, limit: int = 90) -> str:
    """Collapse whitespace and truncate ``text`` for use in reports."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."
