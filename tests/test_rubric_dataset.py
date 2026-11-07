import json

import pytest

from llm_eval_kit import (
    Criterion,
    DatasetError,
    Rubric,
    RubricError,
    Sample,
    load_jsonl,
    load_rubric,
)


def test_load_example_json_rubric(examples_dir):
    rubric = load_rubric(examples_dir / "rubric.json")
    assert rubric.name == "grounded-qa"
    assert [c.id for c in rubric.criteria][:2] == ["groundedness", "consistency"]
    assert rubric.total_weight == 10
    assert rubric.is_blocking(rubric.get("completeness"))
    assert not rubric.is_blocking(rubric.get("tone"))


def test_yaml_rubric_matches_json(examples_dir):
    pytest.importorskip("yaml")
    assert (
        load_rubric(examples_dir / "rubric.yaml").to_dict()
        == load_rubric(examples_dir / "rubric.json").to_dict()
    )


def test_yaml_without_pyyaml_gives_helpful_error(examples_dir, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RubricError, match="PyYAML"):
        load_rubric(examples_dir / "rubric.yaml")


@pytest.mark.parametrize(
    "criterion, message",
    [
        ({"id": "a", "check": "length", "severity": "urgent"}, "severity"),
        ({"id": "a", "check": "length", "weight": -1}, "weight"),
        ({"id": "a", "check": "length", "threshold": 1.5}, "threshold"),
        ({"id": "a"}, "exactly one"),
        ({"id": "a", "check": "length", "judge": "llm"}, "exactly one"),
        ({"id": "a", "check": "length", "wieght": 2}, "unknown keys"),
        ({"check": "length"}, "id"),
    ],
)
def test_invalid_criteria_are_rejected(criterion, message):
    with pytest.raises(RubricError, match=message):
        Criterion.from_dict(criterion)


def test_rubric_level_validation(tmp_path):
    with pytest.raises(RubricError, match="duplicate"):
        Rubric.from_dict({"criteria": [{"id": "a", "check": "length"}] * 2})
    with pytest.raises(RubricError, match="no criteria"):
        Rubric(name="empty", criteria=[])
    bad = tmp_path / "bad.json"
    bad.write_text("{ nope", encoding="utf-8")
    with pytest.raises(RubricError, match="invalid JSON"):
        load_rubric(bad)
    with pytest.raises(RubricError, match="not found"):
        load_rubric(tmp_path / "missing.json")


def test_rubric_roundtrip():
    rubric = Rubric.from_dict(
        {"name": "r", "criteria": [{"id": "a", "judge": "llm", "weight": 2, "params": None}]}
    )
    assert Rubric.from_dict(rubric.to_dict()) == rubric


def test_load_example_dataset(examples_dir):
    samples = load_jsonl(examples_dir / "qa_dataset.jsonl")
    assert len(samples) == 8
    assert all(s.context and s.required_facts for s in samples)


def test_sample_aliases_and_metadata():
    sample = Sample.from_dict(
        {"output": "Hi", "source": "ctx", "facts": ["f"], "model": "m-1", "metadata": {"run": 3}}
    )
    assert sample.id == "sample-1"
    assert sample.answer == "Hi" and sample.context == ["ctx"] and sample.required_facts == ["f"]
    assert sample.metadata == {"run": 3, "model": "m-1"}


def test_dataset_errors(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": "a", "answer": "x"}\n\n{"id": "a", "answer": "y"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate"):
        load_jsonl(path)

    path.write_text('{"id": "a", "answer": "x"}\n{broken\n', encoding="utf-8")
    with pytest.raises(DatasetError, match=r"data.jsonl:2"):
        load_jsonl(path)

    path.write_text(json.dumps({"id": "a", "question": "q"}) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="no answer"):
        load_jsonl(path)

    with pytest.raises(DatasetError, match="context"):
        Sample.from_dict({"answer": "x", "context": 5})
