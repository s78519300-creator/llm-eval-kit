"""Applies a rubric to samples and aggregates the results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .checks import CheckResult
from .dataset import Sample
from .judges import BaseJudge, CheckJudge
from .rubric import Criterion, Rubric, RubricError

_EPS = 1e-9


@dataclass
class CriterionResult:
    """The scored outcome of one criterion for one sample."""

    criterion_id: str
    score: float
    threshold: float
    weight: float
    severity: str
    passed: bool
    blocking: bool
    scorer: str
    justification: str
    evidence: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SampleResult:
    """Overall verdict for a sample, with every criterion result that led to it."""

    sample_id: str
    score: float
    passed: bool
    verdict: str
    criteria: list[CriterionResult]
    question: str = ""
    answer: str = ""

    @property
    def failed(self) -> list[CriterionResult]:
        return [c for c in self.criteria if not c.passed]

    @property
    def blocking_failures(self) -> list[CriterionResult]:
        return [c for c in self.criteria if c.blocking and not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "score": self.score,
            "passed": self.passed,
            "verdict": self.verdict,
            "question": self.question,
            "answer": self.answer,
            "criteria": [c.to_dict() for c in self.criteria],
        }


@dataclass
class RunResult:
    """Results of evaluating a whole dataset against a rubric."""

    rubric: Rubric
    results: list[SampleResult]
    dataset: Optional[str] = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.results) / len(self.results) if self.results else 0.0

    def criterion_stats(self) -> dict[str, dict[str, Any]]:
        """Mean score and pass count for each criterion across all samples."""
        stats: dict[str, dict[str, Any]] = {}
        for criterion in self.rubric.criteria:
            rows = [c for r in self.results for c in r.criteria if c.criterion_id == criterion.id]
            n = len(rows)
            stats[criterion.id] = {
                "weight": criterion.weight,
                "severity": criterion.severity,
                "mean_score": round(sum(c.score for c in rows) / n, 4) if n else 0.0,
                "passed": sum(c.passed for c in rows),
                "total": n,
            }
        return stats

    def summary(self) -> dict[str, Any]:
        return {
            "samples": len(self.results),
            "passed": self.passed,
            "failed": len(self.results) - self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "mean_score": round(self.mean_score, 4),
            "criteria": self.criterion_stats(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric": self.rubric.to_dict(),
            "dataset": self.dataset,
            "started_at": self.started_at,
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }


class Evaluator:
    """Scores samples against a rubric.

    Built-in checks are resolved when the evaluator is created, so a typo in a
    check name or its params fails fast instead of halfway through a dataset.
    Criteria that name a ``judge`` are looked up in the ``judges`` mapping.
    """

    def __init__(self, rubric: Rubric, judges: Optional[Mapping[str, BaseJudge]] = None) -> None:
        self.rubric = rubric
        self.judges = dict(judges or {})
        self._scorers = {c.id: self._resolve(c) for c in rubric.criteria}

    def _resolve(self, criterion: Criterion) -> BaseJudge:
        if criterion.judge:
            try:
                return self.judges[criterion.judge]
            except KeyError:
                raise RubricError(
                    f"criterion {criterion.id!r} uses judge {criterion.judge!r}, "
                    "but no judge with that name was passed to the Evaluator"
                ) from None
        assert criterion.check is not None
        return CheckJudge(criterion.check, criterion.params)

    def score_criterion(self, sample: Sample, criterion: Criterion) -> CriterionResult:
        """Run a single criterion. Judge exceptions are recorded, not raised."""
        error = None
        try:
            result = self._scorers[criterion.id].evaluate(sample, criterion)
            if not isinstance(result, CheckResult):
                raise TypeError(f"judge returned {type(result).__name__}, expected CheckResult")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = CheckResult(0.0, f"Scorer failed ({error}); scored as 0.")
        passed = error is None and result.score + _EPS >= criterion.threshold
        return CriterionResult(
            criterion_id=criterion.id,
            score=round(result.score, 4),
            threshold=criterion.threshold,
            weight=criterion.weight,
            severity=criterion.severity,
            passed=passed,
            blocking=self.rubric.is_blocking(criterion),
            scorer=criterion.scorer,
            justification=result.justification,
            evidence=list(result.evidence),
            error=error,
        )

    def evaluate(self, sample: Sample) -> SampleResult:
        """Score one sample against every criterion and decide pass/fail."""
        rows = [self.score_criterion(sample, c) for c in self.rubric.criteria]
        total_weight = self.rubric.total_weight
        score = sum(r.score * r.weight for r in rows) / total_weight
        blocking = [r.criterion_id for r in rows if r.blocking and not r.passed]
        threshold = self.rubric.pass_threshold

        if blocking:
            passed = False
            verdict = f"Blocking criterion failed: {', '.join(blocking)}."
        elif score + _EPS < threshold:
            passed = False
            verdict = f"Weighted score {score:.2f} is below the pass threshold {threshold:.2f}."
        else:
            passed = True
            soft = [r.criterion_id for r in rows if not r.passed]
            verdict = f"Weighted score {score:.2f} meets the pass threshold {threshold:.2f}."
            if soft:
                verdict += f" Non-blocking issues: {', '.join(soft)}."

        return SampleResult(
            sample_id=sample.id,
            score=round(score, 4),
            passed=passed,
            verdict=verdict,
            criteria=rows,
            question=sample.question,
            answer=sample.answer,
        )

    def run(self, samples: Iterable[Sample], dataset: Optional[str] = None) -> RunResult:
        """Evaluate a collection of samples."""
        return RunResult(self.rubric, [self.evaluate(s) for s in samples], dataset=dataset)
