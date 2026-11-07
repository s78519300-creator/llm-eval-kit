import pytest

from llm_eval_kit.text import (
    content_tokens,
    extract_numbers,
    has_negation,
    jaccard,
    ngrams,
    normalize_number,
    recall,
    shorten,
    split_clauses,
    split_sentences,
    tokenize,
)


def test_tokenize_normalises_numbers_and_case():
    assert tokenize("Revenue rose to 1,200 in 2024, up 3.50%") == [
        "revenue",
        "rose",
        "to",
        "1200",
        "in",
        "2024",
        "up",
        "3.5",
    ]


@pytest.mark.parametrize(
    "raw, expected", [("1,200", "1200"), ("3.50", "3.5"), ("10.0", "10"), ("42", "42")]
)
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


def test_content_tokens_drop_stopwords_and_negations_and_stem_plurals():
    assert content_tokens("The cells are not absorbing glucose molecules") == [
        "cell",
        "absorbing",
        "glucose",
        "molecule",
    ]
    assert "2020" not in content_tokens("in 2020 sales grew", keep_numbers=False)


def test_extract_numbers_ignores_digits_inside_words():
    assert extract_numbers("COVID-19 cases hit 1,500 in 2021; model A4 excluded") == {
        "19",
        "1500",
        "2021",
    }


@pytest.mark.parametrize(
    "text, expected",
    [
        ("It is not available.", True),
        ("It isn't here", True),
        ("It is available.", False),
        ("Never again", True),
    ],
)
def test_has_negation(text, expected):
    assert has_negation(text) is expected


def test_split_sentences_handles_decimals_bullets_and_newlines():
    text = "Pi is about 3.14. It is irrational!\n- First item\n- Second item"
    assert split_sentences(text) == [
        "Pi is about 3.14.",
        "It is irrational!",
        "First item",
        "Second item",
    ]


def test_split_clauses_on_conjunctions():
    assert split_clauses("It runs fast, but it does not scale; costs are high") == [
        "It runs fast",
        "it does not scale",
        "costs are high",
    ]


def test_set_helpers():
    assert ngrams(["a", "b", "c"], 2) == {("a", "b"), ("b", "c")}
    assert recall(["a", "b"], ["a"]) == 0.5
    assert recall([], ["a"]) == 1.0
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        ngrams(["a"], 0)


def test_shorten():
    assert shorten("short") == "short"
    long = shorten("word " * 50, limit=20)
    assert len(long) <= 20 and long.endswith("...")
