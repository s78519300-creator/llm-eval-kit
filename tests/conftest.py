from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval_kit import Rubric, Sample

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES


@pytest.fixture
def eiffel_context() -> list[str]:
    return [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
        "It was completed in 1889 for the World's Fair. "
        "The tower is 330 metres tall, including its antennas."
    ]


@pytest.fixture
def make_sample(eiffel_context):
    def factory(answer: str, **kwargs) -> Sample:
        kwargs.setdefault("context", eiffel_context)
        return Sample(id=kwargs.pop("id", "s1"), answer=answer, **kwargs)

    return factory


@pytest.fixture
def simple_rubric() -> Rubric:
    return Rubric.from_dict(
        {
            "name": "test-rubric",
            "pass_threshold": 0.7,
            "criteria": [
                {
                    "id": "grounded",
                    "check": "hallucination",
                    "weight": 2,
                    "severity": "critical",
                    "threshold": 0.75,
                },
                {
                    "id": "complete",
                    "check": "omission",
                    "weight": 1,
                    "severity": "medium",
                    "threshold": 1.0,
                },
                {
                    "id": "short",
                    "check": "length",
                    "weight": 1,
                    "severity": "low",
                    "threshold": 1.0,
                    "params": {"max_words": 30},
                },
            ],
        }
    )
