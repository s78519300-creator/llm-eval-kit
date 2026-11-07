"""The judge interface: anything that can turn a (sample, criterion) pair into a score.

Built-in checks are wrapped in :class:`CheckJudge`. To plug in your own logic,
including an LLM-as-judge, either subclass :class:`BaseJudge` or wrap a plain
function with :class:`CallableJudge`. :class:`PromptJudge` handles the common
"render a prompt, call a model, parse SCORE/REASON" pattern while leaving the
actual model call to you, so nothing in this package touches the network.
"""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Callable, Optional

from .checks import CheckResult, get_check
from .dataset import Sample
from .rubric import Criterion, RubricError


class BaseJudge(ABC):
    """Abstract base class for every scorer used by the evaluator."""

    name: str = "judge"

    @abstractmethod
    def evaluate(self, sample: Sample, criterion: Criterion) -> CheckResult:
        """Score ``sample`` against ``criterion`` and explain the score."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class CheckJudge(BaseJudge):
    """Runs one of the registered deterministic checks with fixed parameters."""

    def __init__(self, check: str, params: Optional[Mapping[str, Any]] = None) -> None:
        try:
            self._fn = get_check(check)
        except KeyError as exc:
            raise RubricError(str(exc.args[0])) from None
        self.name = check
        self.params = dict(params or {})
        try:
            inspect.signature(self._fn).bind(None, **self.params)
        except TypeError as exc:
            raise RubricError(f"invalid params for check {check!r}: {exc}") from None

    def evaluate(self, sample: Sample, criterion: Criterion) -> CheckResult:
        return self._fn(sample, **self.params)


JudgeFn = Callable[[Sample, Criterion], Any]


class CallableJudge(BaseJudge):
    """Adapts any function ``fn(sample, criterion)`` into a judge.

    The function may return a :class:`CheckResult`, a bare score, a
    ``(score, justification)`` tuple, a bool, or a dict with ``score`` and an
    optional ``justification``/``reason`` and ``evidence``.
    """

    def __init__(self, fn: JudgeFn, name: Optional[str] = None) -> None:
        if not callable(fn):
            raise TypeError("CallableJudge needs a callable")
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "callable")

    def evaluate(self, sample: Sample, criterion: Criterion) -> CheckResult:
        return coerce_result(self._fn(sample, criterion), source=self.name)


def coerce_result(value: Any, source: str = "judge") -> CheckResult:
    """Normalise the many shapes a custom judge might return into a CheckResult."""
    if isinstance(value, CheckResult):
        return value
    if isinstance(value, bool):
        return CheckResult(float(value), f"{source} returned {value}.")
    if isinstance(value, (int, float)):
        return CheckResult(float(value), f"{source} returned score {float(value):.2f}.")
    if isinstance(value, tuple) and len(value) == 2:
        score, reason = value
        return CheckResult(float(score), str(reason))
    if isinstance(value, Mapping) and "score" in value:
        reason = value.get("justification") or value.get("reason") or f"{source} returned a score."
        return CheckResult(float(value["score"]), str(reason), list(value.get("evidence", [])))
    raise TypeError(f"{source} returned unsupported value of type {type(value).__name__}")


DEFAULT_PROMPT = """\
Evaluate the ANSWER against the CRITERION using only the CONTEXT as ground truth.

CRITERION: {criterion}
QUESTION: {question}

CONTEXT:
{context}

ANSWER:
{answer}

Reply with exactly two lines:
SCORE: <a number from 0 to 1>
REASON: <one sentence explaining the score>
"""

_SCORE_RE = re.compile(r"score\s*[:=]\s*(-?\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?", re.I)
_REASON_RE = re.compile(r"reason\s*[:=]\s*(.+)", re.I)


def parse_judgement(text: str, scale: float = 1.0) -> tuple[float, str]:
    """Extract ``(score, reason)`` from a model reply such as ``SCORE: 4/5``.

    A fractional score (``4/5``) is normalised by its denominator; otherwise
    the raw number is divided by ``scale``.
    """
    match = _SCORE_RE.search(text)
    if not match:
        raise ValueError(f"no SCORE found in judge reply: {text[:80]!r}")
    raw = float(match.group(1))
    score = raw / float(match.group(2)) if match.group(2) else raw / scale
    reason_match = _REASON_RE.search(text)
    reason = reason_match.group(1).strip() if reason_match else "No reason given."
    return max(0.0, min(1.0, score)), reason


class PromptJudge(BaseJudge):
    """A model-agnostic LLM-as-judge.

    ``complete`` is any function that takes a prompt string and returns the
    model's reply: an SDK call, a local model, or a stub in tests. The judge
    only formats the prompt and parses the reply.
    """

    def __init__(
        self,
        complete: Callable[[str], str],
        template: str = DEFAULT_PROMPT,
        scale: float = 1.0,
        name: str = "prompt-judge",
    ) -> None:
        self._complete = complete
        self.template = template
        self.scale = scale
        self.name = name

    def build_prompt(self, sample: Sample, criterion: Criterion) -> str:
        """Render the prompt for one sample/criterion pair."""
        return self.template.format(
            criterion=criterion.description or criterion.id,
            question=sample.question or "(none)",
            context=sample.context_text or "(none)",
            answer=sample.answer,
        )

    def evaluate(self, sample: Sample, criterion: Criterion) -> CheckResult:
        reply = self._complete(self.build_prompt(sample, criterion))
        score, reason = parse_judgement(reply, self.scale)
        return CheckResult(score, reason, details={"raw_reply": reply})
