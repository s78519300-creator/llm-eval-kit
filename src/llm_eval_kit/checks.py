"""Built-in deterministic checks.

Every check is a plain function ``check(sample, **params) -> CheckResult`` that
is registered under a short name with :func:`register_check`. Checks never call
out to a model or the network, so the same input always produces the same
score and justification, which is what makes them useful as regression tests.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .dataset import Sample
from .text import (
    RSQUO,
    content_tokens,
    extract_numbers,
    has_negation,
    jaccard,
    ngrams,
    recall,
    shorten,
    split_clauses,
    split_sentences,
)


@dataclass
class CheckResult:
    """Outcome of a single check: a score in ``[0, 1]`` plus the reasoning behind it."""

    score: float
    justification: str
    evidence: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, float(self.score)))


CheckFn = Callable[..., CheckResult]
_REGISTRY: dict[str, CheckFn] = {}


def register_check(name: str) -> Callable[[CheckFn], CheckFn]:
    """Decorator that makes a check available to rubrics under ``name``."""

    def decorator(fn: CheckFn) -> CheckFn:
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise ValueError(f"a check named {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_check(name: str) -> CheckFn:
    """Look up a registered check by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown check {name!r} (available: {known})") from None


def available_checks() -> dict[str, str]:
    """Map each registered check name to the first line of its docstring."""
    return {
        name: (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        for name, fn in sorted(_REGISTRY.items())
    }


def _quote(text: str) -> str:
    return f'"{shorten(text)}"'


# --------------------------------------------------------------------------- #
# Grounding checks
# --------------------------------------------------------------------------- #


def claim_support(claim: str, context_tokens: Sequence[str], n: int = 2) -> float:
    """Score how well ``claim`` is supported by the context tokens.

    The score blends unigram recall (are the claim's content words present at
    all?) with n-gram recall (do they appear in the same order/phrasing?). A
    claim that reuses source vocabulary but recombines it into a new statement
    scores lower than one that paraphrases closely.
    """
    tokens = content_tokens(claim)
    if not tokens:
        return 1.0
    unigram = recall(tokens, context_tokens)
    if len(tokens) < n:
        return unigram
    phrase = recall(ngrams(tokens, n), ngrams(context_tokens, n))
    return 0.5 * unigram + 0.5 * phrase


@register_check("hallucination")
def hallucination(
    sample: Sample,
    min_support: float = 0.6,
    ngram: int = 2,
    strict_numbers: bool = True,
) -> CheckResult:
    """Flag answer sentences that are not supported by the source context.

    Each sentence of the answer is treated as a claim and scored with
    :func:`claim_support`. With ``strict_numbers`` a claim that mentions a number
    absent from the context is always unsupported. The check score is the
    fraction of claims that are supported.
    """
    claims = [s for s in split_sentences(sample.answer) if content_tokens(s)]
    if not claims:
        return CheckResult(1.0, "Answer contains no checkable claims.")
    if not sample.context:
        return CheckResult(0.0, "No source context was provided, so no claim can be verified.")

    ctx_tokens = content_tokens(sample.context_text)
    ctx_numbers = extract_numbers(sample.context_text)
    supported = 0
    evidence = []
    per_claim = []
    for claim in claims:
        support = claim_support(claim, ctx_tokens, ngram)
        unknown_numbers = sorted(extract_numbers(claim) - ctx_numbers) if strict_numbers else []
        ok = support >= min_support and not unknown_numbers
        per_claim.append({"claim": claim, "support": round(support, 3), "supported": ok})
        if ok:
            supported += 1
            continue
        reason = f"support {support:.2f}"
        if unknown_numbers:
            reason += f", numbers not in context: {', '.join(unknown_numbers)}"
        evidence.append(f"Unsupported: {_quote(claim)} ({reason})")

    total = len(claims)
    if supported == total:
        why = f"All {total} claim(s) are supported by the context (min support {min_support:.2f})."
    else:
        why = (
            f"{total - supported} of {total} claim(s) lack support in the context "
            f"(min support {min_support:.2f})."
        )
    return CheckResult(supported / total, why, evidence, {"claims": per_claim})


@register_check("omission")
def omission(
    sample: Sample,
    facts: Optional[Sequence[str]] = None,
    min_coverage: float = 0.75,
) -> CheckResult:
    """Check that the answer mentions every required key fact.

    Facts come from ``params.facts`` or, by default, the sample's
    ``required_facts``. A fact counts as present when enough of its content
    words appear in the answer and every number it mentions appears verbatim.
    """
    required = list(facts) if facts is not None else list(sample.required_facts)
    if not required:
        return CheckResult(1.0, "No required facts defined for this sample.")

    answer_tokens = set(content_tokens(sample.answer))
    answer_numbers = extract_numbers(sample.answer)
    missing = []
    for fact in required:
        coverage = recall(content_tokens(fact), answer_tokens)
        numbers_ok = extract_numbers(fact) <= answer_numbers
        if coverage < min_coverage or not numbers_ok:
            missing.append((fact, coverage))

    found = len(required) - len(missing)
    evidence = [f"Missing: {_quote(f)} (coverage {c:.2f})" for f, c in missing]
    if missing:
        why = f"{len(missing)} of {len(required)} required fact(s) missing from the answer."
    else:
        why = f"All {len(required)} required fact(s) are present."
    return CheckResult(found / len(required), why, evidence, {"missing": [f for f, _ in missing]})


@register_check("contradiction")
def contradiction(sample: Sample, min_alignment: float = 0.5) -> CheckResult:
    """Detect answer statements that contradict the context they paraphrase.

    Each answer clause is aligned to the context sentences that share most of
    its vocabulary. A contradiction is reported when the closest sentence has
    the opposite polarity (one is negated, the other is not), or when the
    answer states a number that none of the aligned sentences contain while
    they do state a different one.
    """
    ctx_sentences = [s for c in sample.context for s in split_sentences(c)]
    if not ctx_sentences:
        return CheckResult(1.0, "No context provided; nothing to contradict.")
    ctx_terms = [set(content_tokens(s, keep_numbers=False)) for s in ctx_sentences]

    checked = 0
    issues: list[str] = []
    claims = [c for s in split_sentences(sample.answer) for c in split_clauses(s)]
    for claim in claims:
        terms = set(content_tokens(claim, keep_numbers=False))
        if len(terms) < 2:
            continue
        aligned = [i for i, ct in enumerate(ctx_terms) if ct and recall(terms, ct) >= min_alignment]
        if not aligned:
            continue
        checked += 1
        best = max(aligned, key=lambda i: jaccard(terms, ctx_terms[i]))
        source = ctx_sentences[best]

        if has_negation(claim) != has_negation(source):
            issues.append(f"Polarity conflict: {_quote(claim)} vs source {_quote(source)}")
            continue

        claim_numbers = extract_numbers(claim)
        source_numbers: set[str] = set()
        for i in aligned:
            source_numbers |= extract_numbers(ctx_sentences[i])
        conflicting = sorted(claim_numbers - source_numbers)
        if conflicting and source_numbers:
            stated = ", ".join(sorted(source_numbers))
            issues.append(
                f"Number conflict ({', '.join(conflicting)} vs {stated}): "
                f"{_quote(claim)} vs source {_quote(source)}"
            )

    if not checked:
        return CheckResult(1.0, "No answer clause overlaps the context closely enough to compare.")
    score = 1.0 - len(issues) / checked
    if issues:
        why = f"{len(issues)} of {checked} aligned clause(s) contradict the context."
    else:
        why = f"No contradictions found across {checked} aligned clause(s)."
    return CheckResult(score, why, issues, {"checked": checked})


# --------------------------------------------------------------------------- #
# Format checks
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


@register_check("json_valid")
def json_valid(sample: Sample, required_keys: Optional[Sequence[str]] = None) -> CheckResult:
    """Check that the answer is valid JSON, optionally with required top-level keys.

    A surrounding Markdown code fence is tolerated. Missing keys reduce the
    score proportionally.
    """
    raw = sample.answer.strip()
    fenced = _FENCE_RE.match(raw)
    if fenced:
        raw = fenced.group(1)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return CheckResult(0.0, f"Answer is not valid JSON: {exc.msg} at char {exc.pos}.")

    keys = list(required_keys or [])
    if not keys:
        return CheckResult(1.0, "Answer is valid JSON.")
    if not isinstance(payload, dict):
        return CheckResult(
            0.0, f"Expected a JSON object with keys {keys}, got {type(payload).__name__}."
        )
    missing = [k for k in keys if k not in payload]
    if missing:
        return CheckResult(
            (len(keys) - len(missing)) / len(keys),
            f"Valid JSON but missing {len(missing)} of {len(keys)} required key(s).",
            [f"Missing key: {k}" for k in missing],
        )
    return CheckResult(1.0, f"Valid JSON with all {len(keys)} required key(s).")


@register_check("length")
def length(
    sample: Sample,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> CheckResult:
    """Check that the answer length is within word and/or character bounds.

    Out-of-range answers are scored proportionally (an answer twice the
    maximum length scores 0.5) so small overruns are not punished like large ones.
    """
    words = len(sample.answer.split())
    chars = len(sample.answer)
    score = 1.0
    problems = []
    for value, low, high, unit in (
        (words, min_words, max_words, "words"),
        (chars, min_chars, max_chars, "chars"),
    ):
        if low is not None and value < low:
            score = min(score, value / low if low else 1.0)
            problems.append(f"{value} {unit} is below the minimum of {low}")
        if high is not None and value > high:
            score = min(score, high / value)
            problems.append(f"{value} {unit} exceeds the maximum of {high}")
    if problems:
        return CheckResult(score, "Length out of bounds: " + "; ".join(problems) + ".", problems)
    return CheckResult(1.0, f"Length OK ({words} words, {chars} chars).")


def _phrase_pattern(phrase: str, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    escaped = re.escape(phrase).replace("'", f"['{RSQUO}]")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags)


@register_check("required_keywords")
def required_keywords(
    sample: Sample, keywords: Sequence[str] = (), case_sensitive: bool = False
) -> CheckResult:
    """Check that the answer contains each of the given keywords or phrases."""
    if not keywords:
        return CheckResult(1.0, "No required keywords configured.")
    missing = [k for k in keywords if not _phrase_pattern(k, case_sensitive).search(sample.answer)]
    found = len(keywords) - len(missing)
    if missing:
        return CheckResult(
            found / len(keywords),
            f"{len(missing)} of {len(keywords)} required keyword(s) missing.",
            [f"Missing keyword: {k!r}" for k in missing],
        )
    return CheckResult(1.0, f"All {len(keywords)} required keyword(s) present.")


@register_check("forbidden_phrases")
def forbidden_phrases(
    sample: Sample, phrases: Sequence[str] = (), case_sensitive: bool = False
) -> CheckResult:
    """Fail the answer if it contains any forbidden phrase (hedging, boilerplate, etc.)."""
    hits = [p for p in phrases if _phrase_pattern(p, case_sensitive).search(sample.answer)]
    if hits:
        return CheckResult(
            0.0,
            f"Answer contains {len(hits)} forbidden phrase(s).",
            [f"Forbidden phrase: {p!r}" for p in hits],
        )
    return CheckResult(1.0, "No forbidden phrases found.")
