import pytest

from llm_eval_kit import (
    BaseJudge,
    CallableJudge,
    CheckJudge,
    CheckResult,
    Criterion,
    PromptJudge,
    RubricError,
    Sample,
    parse_judgement,
)
from llm_eval_kit.judges import coerce_result

CRITERION = Criterion(id="c", judge="j", description="Be accurate")
SAMPLE = Sample(id="s", question="Q?", answer="An answer.", context=["Some context."])


def test_base_judge_is_abstract():
    with pytest.raises(TypeError):
        BaseJudge()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "value, score, reason",
    [
        (0.25, 0.25, "returned score 0.25"),
        (True, 1.0, "returned True"),
        ((0.5, "half right"), 0.5, "half right"),
        ({"score": 0.9, "reason": "good", "evidence": ["e1"]}, 0.9, "good"),
        (CheckResult(0.3, "as is"), 0.3, "as is"),
    ],
)
def test_coerce_result_shapes(value, score, reason):
    result = coerce_result(value, source="fn")
    assert result.score == score
    assert reason in result.justification


def test_coerce_result_rejects_unknown_type():
    with pytest.raises(TypeError):
        coerce_result("great!")


def test_callable_judge_receives_sample_and_criterion():
    seen = {}

    def judge(sample, criterion):
        seen["args"] = (sample.id, criterion.id)
        return 1.0

    cj = CallableJudge(judge)
    assert cj.name == "judge"
    assert cj.evaluate(SAMPLE, CRITERION).score == 1.0
    assert seen["args"] == ("s", "c")


def test_check_judge_validates_name_and_params():
    with pytest.raises(RubricError, match="unknown check"):
        CheckJudge("nope")
    with pytest.raises(RubricError, match="invalid params"):
        CheckJudge("length", {"max_wordz": 5})
    assert CheckJudge("length", {"max_words": 5}).evaluate(SAMPLE, CRITERION).score == 1.0


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("SCORE: 0.8\nREASON: mostly right", (0.8, "mostly right")),
        ("Score = 4/5\nReason: fine", (0.8, "fine")),
        ("score: 7", (0.7, "No reason given.")),
    ],
)
def test_parse_judgement(reply, expected):
    scale = 10 if reply == "score: 7" else 1
    assert parse_judgement(reply, scale=scale) == pytest.approx(expected)


def test_parse_judgement_requires_score():
    with pytest.raises(ValueError):
        parse_judgement("Looks good to me")


def test_prompt_judge_formats_prompt_and_parses_reply():
    prompts = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return "SCORE: 3/4\nREASON: Accurate but terse."

    result = PromptJudge(complete).evaluate(SAMPLE, CRITERION)
    assert result.score == 0.75
    assert result.justification == "Accurate but terse."
    assert "CRITERION: Be accurate" in prompts[0]
    assert "Some context." in prompts[0] and "An answer." in prompts[0]
