"""Evaluation samples and JSONL dataset loading."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]

# Field aliases make it easy to point the tool at data exported from other systems.
_ANSWER_KEYS = ("answer", "output", "response", "prediction")
_CONTEXT_KEYS = ("context", "contexts", "source", "sources", "reference")
_FACT_KEYS = ("required_facts", "key_facts", "facts")


class DatasetError(ValueError):
    """Raised when a dataset file or record is malformed."""


@dataclass
class Sample:
    """A single model output to be evaluated, plus the material to judge it against."""

    id: str
    answer: str
    question: str = ""
    context: list[str] = field(default_factory=list)
    required_facts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_text(self) -> str:
        """All context passages joined into one string."""
        return "\n".join(self.context)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], index: int = 0) -> Sample:
        """Build a sample from a mapping, accepting a few common field aliases."""
        if not isinstance(data, Mapping):
            raise DatasetError(f"record {index} must be a JSON object, got {type(data).__name__}")

        answer = _first(data, _ANSWER_KEYS)
        if answer is None:
            raise DatasetError(f"record {index} has no answer (expected one of {_ANSWER_KEYS})")
        if not isinstance(answer, str):
            raise DatasetError(f"record {index}: answer must be a string")

        context = _as_str_list(_first(data, _CONTEXT_KEYS), "context", index)
        facts = _as_str_list(_first(data, _FACT_KEYS), "required_facts", index)
        known = {"id", "question", *_ANSWER_KEYS, *_CONTEXT_KEYS, *_FACT_KEYS, "metadata"}
        metadata = dict(data.get("metadata") or {})
        metadata.update({k: v for k, v in data.items() if k not in known})

        return cls(
            id=str(data.get("id", f"sample-{index + 1}")),
            answer=answer,
            question=str(data.get("question", "")),
            context=context,
            required_facts=facts,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the sample back to a plain dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "context": list(self.context),
            "required_facts": list(self.required_facts),
            "metadata": dict(self.metadata),
        }


def _first(data: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _as_str_list(value: Any, name: str, index: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise DatasetError(f"record {index}: {name} must be a string or a list of strings")


def iter_jsonl(path: PathLike) -> Iterator[Sample]:
    """Yield samples from a JSON Lines file, skipping blank lines."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")
    with path.open(encoding="utf-8") as handle:
        index = 0
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            yield Sample.from_dict(record, index)
            index += 1


def load_jsonl(path: PathLike) -> list[Sample]:
    """Load every sample from a JSON Lines file and check that ids are unique."""
    samples = list(iter_jsonl(path))
    seen: set[str] = set()
    for sample in samples:
        if sample.id in seen:
            raise DatasetError(f"duplicate sample id: {sample.id!r}")
        seen.add(sample.id)
    return samples
