"""Rubric and criterion definitions, plus loaders for JSON and YAML files."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

SEVERITIES = ("low", "medium", "high", "critical")
DEFAULT_BLOCKING = ("high", "critical")


class RubricError(ValueError):
    """Raised when a rubric is invalid or refers to something that does not exist."""


@dataclass
class Criterion:
    """One thing the rubric measures.

    A criterion is scored either by a built-in deterministic ``check`` or by a
    named ``judge`` supplied to the evaluator at runtime (for example an
    LLM-as-judge). Scores are always in ``[0, 1]`` and the criterion passes when
    the score reaches ``threshold``.
    """

    id: str
    description: str = ""
    weight: float = 1.0
    severity: str = "medium"
    threshold: float = 1.0
    check: Optional[str] = None
    judge: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise RubricError("every criterion needs a non-empty string 'id'")
        if self.severity not in SEVERITIES:
            raise RubricError(
                f"criterion {self.id!r}: severity must be one of {SEVERITIES}, "
                f"got {self.severity!r}"
            )
        if not isinstance(self.weight, (int, float)) or self.weight < 0:
            raise RubricError(f"criterion {self.id!r}: weight must be a non-negative number")
        if not isinstance(self.threshold, (int, float)) or not 0 <= self.threshold <= 1:
            raise RubricError(f"criterion {self.id!r}: threshold must be between 0 and 1")
        if bool(self.check) == bool(self.judge):
            raise RubricError(f"criterion {self.id!r}: set exactly one of 'check' or 'judge'")
        if not isinstance(self.params, dict):
            raise RubricError(f"criterion {self.id!r}: params must be a mapping")

    @property
    def scorer(self) -> str:
        """Human-readable name of whatever produces this criterion's score."""
        return f"check:{self.check}" if self.check else f"judge:{self.judge}"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Criterion:
        """Build a criterion from a mapping, rejecting unknown keys to catch typos early."""
        if not isinstance(data, Mapping):
            raise RubricError("each criterion must be a mapping")
        allowed = {
            "id",
            "description",
            "weight",
            "severity",
            "threshold",
            "check",
            "judge",
            "params",
        }
        unknown = set(data) - allowed
        if unknown:
            raise RubricError(f"criterion {data.get('id')!r}: unknown keys {sorted(unknown)}")
        if "id" not in data:
            raise RubricError("every criterion needs a non-empty string 'id'")
        kwargs = dict(data)
        if kwargs.get("params") is None:
            kwargs.pop("params", None)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (omits whichever scorer is unused)."""
        out: dict[str, Any] = {
            "id": self.id,
            "description": self.description,
            "weight": self.weight,
            "severity": self.severity,
            "threshold": self.threshold,
        }
        if self.check:
            out["check"] = self.check
        if self.judge:
            out["judge"] = self.judge
        if self.params:
            out["params"] = dict(self.params)
        return out


@dataclass
class Rubric:
    """A named, versioned set of weighted criteria with an overall pass threshold."""

    name: str
    criteria: list[Criterion]
    description: str = ""
    version: str = "1.0"
    pass_threshold: float = 0.8
    blocking_severities: Sequence[str] = DEFAULT_BLOCKING

    def __post_init__(self) -> None:
        if not self.criteria:
            raise RubricError(f"rubric {self.name!r} has no criteria")
        ids = [c.id for c in self.criteria]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise RubricError(f"rubric {self.name!r}: duplicate criterion ids {dupes}")
        if not 0 <= self.pass_threshold <= 1:
            raise RubricError("pass_threshold must be between 0 and 1")
        bad = [s for s in self.blocking_severities if s not in SEVERITIES]
        if bad:
            raise RubricError(f"blocking_severities contains unknown values {bad}")
        if sum(c.weight for c in self.criteria) <= 0:
            raise RubricError("at least one criterion must have a positive weight")
        self.blocking_severities = tuple(self.blocking_severities)

    @property
    def total_weight(self) -> float:
        return float(sum(c.weight for c in self.criteria))

    def get(self, criterion_id: str) -> Criterion:
        """Return a criterion by id."""
        for criterion in self.criteria:
            if criterion.id == criterion_id:
                return criterion
        raise KeyError(criterion_id)

    def is_blocking(self, criterion: Criterion) -> bool:
        """Whether failing ``criterion`` fails the whole sample regardless of score."""
        return criterion.severity in self.blocking_severities

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Rubric:
        """Build a rubric from a parsed JSON/YAML document."""
        if not isinstance(data, Mapping):
            raise RubricError("rubric document must be a mapping at the top level")
        raw_criteria = data.get("criteria")
        if not isinstance(raw_criteria, list):
            raise RubricError("rubric needs a 'criteria' list")
        return cls(
            name=str(data.get("name", "unnamed-rubric")),
            description=str(data.get("description", "")),
            version=str(data.get("version", "1.0")),
            pass_threshold=float(data.get("pass_threshold", 0.8)),
            blocking_severities=tuple(data.get("blocking_severities", DEFAULT_BLOCKING)),
            criteria=[Criterion.from_dict(c) for c in raw_criteria],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "pass_threshold": self.pass_threshold,
            "blocking_severities": list(self.blocking_severities),
            "criteria": [c.to_dict() for c in self.criteria],
        }


def load_rubric(path: Union[str, Path]) -> Rubric:
    """Load a rubric from a ``.json``, ``.yaml`` or ``.yml`` file.

    YAML support needs PyYAML (``pip install llm-eval-kit[yaml]``); JSON always works.
    """
    path = Path(path)
    if not path.is_file():
        raise RubricError(f"rubric not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RubricError(
                "YAML rubrics need PyYAML: pip install 'llm-eval-kit[yaml]' (or use JSON)"
            ) from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise RubricError(f"{path}: invalid YAML ({exc})") from exc
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RubricError(f"{path}: invalid JSON ({exc.msg} at line {exc.lineno})") from exc

    return Rubric.from_dict(data)
