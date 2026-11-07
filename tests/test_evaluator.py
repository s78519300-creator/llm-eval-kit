import pytest

from llm_eval_kit import CallableJudge, Evaluator, Rubric, RubricError, load_jsonl, load_rubric


def test_good_answer_passes(simple_rubric, make_sample):
    sample = make_sample(
        "The Eiffel Tower was completed in 1889.", required_facts=["completed in 1889"]
    )
    result = Evaluator(simple_rubric).evaluate(sample)
    assert result.passed
    assert result.score == 1.0
    assert result.failed == []
    assert "meets the pass threshold" in result.verdict


def test_weighted_score_and_soft_failure(simple_rubric, make_sample):
    # grounded (w=2) passes, omission (w=1) scores 0.5, length (w=1) passes -> 3.5 / 4
    sample = make_sample(
        "The Eiffel Tower was completed in 1889.",
        required_facts=["completed in 1889", "330 metres tall"],
    )
    result = Evaluator(simple_rubric).evaluate(sample)
    assert result.score == pytest.approx(0.875)
    assert result.passed  # 'complete' is medium severity, so not blocking
    assert [c.criterion_id for c in result.failed] == ["complete"]
    assert "Non-blocking issues: complete" in result.verdict


def test_blocking_failure_overrides_score(simple_rubric, make_sample):
    sample = make_sample(
        "The Eiffel Tower was completed in 1889. It was moved to London in 1950 by barge."
    )
    result = Evaluator(simple_rubric).evaluate(sample)
    assert not result.passed
    assert result.blocking_failures[0].criterion_id == "grounded"
    assert result.verdict.startswith("Blocking criterion failed: grounded")


def test_score_below_threshold_fails(make_sample):
    rubric = Rubric.from_dict(
        {
            "name": "r",
            "pass_threshold": 0.9,
            "criteria": [
                {"id": "short", "check": "length", "severity": "low", "params": {"max_words": 3}},
                {
                    "id": "kw",
                    "check": "required_keywords",
                    "severity": "low",
                    "params": {"keywords": ["tower"]},
                },
            ],
        }
    )
    result = Evaluator(rubric).evaluate(make_sample("The tower is tall and old."))
    assert not result.passed
    assert "below the pass threshold" in result.verdict


def test_custom_judge_and_error_isolation(make_sample):
    rubric = Rubric.from_dict(
        {
            "name": "r",
            "criteria": [
                {"id": "ok", "judge": "always", "severity": "low"},
                {"id": "boom", "judge": "broken", "severity": "low", "threshold": 0.0},
            ],
        }
    )

    def broken(sample, criterion):
        raise RuntimeError("model timeout")

    judges = {"always": CallableJudge(lambda s, c: 1.0), "broken": CallableJudge(broken)}
    result = Evaluator(rubric, judges).evaluate(make_sample("Anything."))
    boom = result.criteria[1]
    assert boom.error == "RuntimeError: model timeout"
    assert boom.passed is False and boom.score == 0.0
    assert result.criteria[0].scorer == "judge:always"


def test_missing_judge_fails_fast():
    rubric = Rubric.from_dict({"name": "r", "criteria": [{"id": "a", "judge": "llm"}]})
    with pytest.raises(RubricError, match="no judge with that name"):
        Evaluator(rubric)


def test_example_dataset_end_to_end(examples_dir):
    rubric = load_rubric(examples_dir / "rubric.json")
    run = Evaluator(rubric).run(load_jsonl(examples_dir / "qa_dataset.jsonl"), dataset="qa")
    outcomes = {r.sample_id: r for r in run.results}

    assert {sid for sid, r in outcomes.items() if r.passed} == {
        "eiffel-tower",
        "python-gil",
        "boiling-point",
    }
    failed = {sid: {c.criterion_id for c in r.failed} for sid, r in outcomes.items()}
    assert failed["gift-card-refund"] == {"consistency"}
    assert failed["photosynthesis"] == {"completeness"}
    assert failed["insulin"] == {"groundedness"}
    assert {"groundedness", "consistency"} <= failed["great-wall"]
    assert "consistency" in failed["headphone-battery"]
    assert failed["boiling-point"] == {"tone"}

    summary = run.summary()
    assert summary["samples"] == 8 and summary["passed"] == 3
    assert summary["criteria"]["conciseness"]["passed"] == 8
