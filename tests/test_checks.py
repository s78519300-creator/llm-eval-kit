import pytest

from llm_eval_kit import CheckResult, Sample, available_checks, get_check, register_check
from llm_eval_kit.checks import (
    claim_support,
    contradiction,
    forbidden_phrases,
    hallucination,
    json_valid,
    length,
    omission,
    required_keywords,
)
from llm_eval_kit.text import content_tokens

# --- hallucination -------------------------------------------------------------


def test_hallucination_passes_faithful_answer(make_sample):
    sample = make_sample("The Eiffel Tower was completed in 1889. It is 330 metres tall.")
    result = hallucination(sample)
    assert result.score == 1.0
    assert result.evidence == []


def test_hallucination_flags_unsupported_claim(make_sample):
    sample = make_sample(
        "The Eiffel Tower was completed in 1889. Napoleon personally painted it gold every winter."
    )
    result = hallucination(sample)
    assert result.score == 0.5
    assert "Napoleon" in result.evidence[0]
    assert result.details["claims"][1]["supported"] is False


def test_hallucination_strict_numbers(make_sample):
    sample = make_sample("The Eiffel Tower was completed in 1899 for the World's Fair.")
    assert hallucination(sample).score == 0.0
    assert hallucination(sample, strict_numbers=False).score == 1.0


def test_hallucination_without_context_scores_zero():
    result = hallucination(Sample(id="x", answer="Paris is in France."))
    assert result.score == 0.0
    assert "No source context" in result.justification


def test_claim_support_rewards_phrasing_overlap():
    ctx = content_tokens("The tower is 330 metres tall including its antennas")
    assert claim_support("The tower is 330 metres tall", ctx) == 1.0
    assert claim_support("Antennas tall including tower", ctx) < 0.75


# --- omission -------------------------------------------------------------------


def test_omission_reports_missing_facts(make_sample):
    sample = make_sample(
        "It was completed in 1889.",
        required_facts=["completed in 1889", "330 metres tall"],
    )
    result = omission(sample)
    assert result.score == 0.5
    assert result.details["missing"] == ["330 metres tall"]


def test_omission_requires_exact_numbers(make_sample):
    sample = make_sample("It is 300 metres tall.")
    assert omission(sample, facts=["330 metres tall"]).score == 0.0


def test_omission_with_no_facts_is_neutral(make_sample):
    assert omission(make_sample("Anything.")).score == 1.0


# --- contradiction --------------------------------------------------------------


def test_contradiction_detects_negation_flip():
    sample = Sample(
        id="x",
        context=["Refunds are not available for digital gift cards."],
        answer="Refunds are available for digital gift cards.",
    )
    result = contradiction(sample)
    assert result.score == 0.0
    assert result.evidence[0].startswith("Polarity conflict")


def test_contradiction_detects_number_mismatch(make_sample):
    result = contradiction(make_sample("The tower was completed in 1899 for the World's Fair."))
    assert result.score == 0.0
    assert "1899" in result.evidence[0]


def test_contradiction_checks_compound_sentences_clause_by_clause():
    sample = Sample(
        id="x",
        context=[
            "The GIL allows only one thread to execute bytecode at a time.",
            "CPU-bound programs do not run faster with multiple threads.",
        ],
        answer="The GIL allows only one thread to execute bytecode at a time, "
        "so CPU-bound programs do not run faster with multiple threads.",
    )
    assert contradiction(sample).score == 1.0


def test_contradiction_ignores_unrelated_sentences(make_sample):
    result = contradiction(make_sample("Bananas are rich in potassium."))
    assert result.score == 1.0
    assert "No answer clause overlaps" in result.justification


# --- format checks --------------------------------------------------------------


def test_json_valid_with_fence_and_required_keys():
    good = Sample(id="x", answer='```json\n{"name": "Ada", "age": 36}\n```')
    assert json_valid(good, required_keys=["name", "age"]).score == 1.0
    partial = json_valid(good, required_keys=["name", "email"])
    assert partial.score == 0.5
    assert partial.evidence == ["Missing key: email"]
    assert json_valid(Sample(id="y", answer="{not json")).score == 0.0


def test_length_scores_proportionally():
    sample = Sample(id="x", answer=" ".join(["word"] * 20))
    assert length(sample, max_words=20).score == 1.0
    assert length(sample, max_words=10).score == 0.5
    assert length(sample, min_words=40).score == 0.5
    assert length(sample, max_chars=10).score < 1.0


def test_keyword_and_forbidden_phrase_checks():
    sample = Sample(id="x", answer="I think the Refund window is 30 days.")
    assert required_keywords(sample, keywords=["refund", "30 days"]).score == 1.0
    assert required_keywords(sample, keywords=["refund", "receipt"]).score == 0.5
    assert required_keywords(sample, keywords=["refund"], case_sensitive=True).score == 0.0
    assert forbidden_phrases(sample, phrases=["I think"]).score == 0.0
    assert forbidden_phrases(sample, phrases=["think the r"]).score == 1.0  # word boundaries


def test_forbidden_phrases_match_curly_apostrophes():
    sample = Sample(id="x", answer="I\N{RIGHT SINGLE QUOTATION MARK}m not sure, honestly.")
    assert forbidden_phrases(sample, phrases=["I'm not sure"]).score == 0.0


# --- registry -------------------------------------------------------------------


def test_registry_lists_builtins_and_accepts_custom_checks():
    names = set(available_checks())
    assert {"hallucination", "omission", "contradiction", "json_valid", "length"} <= names

    @register_check("mentions_paris")
    def mentions_paris(sample: Sample) -> CheckResult:
        """Answer mentions Paris."""
        return CheckResult(float("paris" in sample.answer.lower()), "checked for Paris")

    assert get_check("mentions_paris") is mentions_paris
    with pytest.raises(ValueError):
        register_check("mentions_paris")(lambda sample: CheckResult(1, ""))
    with pytest.raises(KeyError, match="unknown check"):
        get_check("does_not_exist")


def test_check_result_clamps_score():
    assert CheckResult(1.7, "").score == 1.0
    assert CheckResult(-2, "").score == 0.0
