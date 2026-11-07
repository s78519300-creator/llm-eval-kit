"""Extending a rubric with custom judges.

Run from the repository root:

    python examples/custom_judge.py

Two extra criteria are added to the example rubric:

* ``answers_question`` uses a ``CallableJudge`` wrapping a plain function.
* ``helpfulness`` uses a ``PromptJudge``. Here its ``complete`` function is a
  local stub so the example runs offline; in practice you would pass a function
  that sends the prompt to your model provider and returns the reply text.
"""

from __future__ import annotations

from pathlib import Path

from llm_eval_kit import (
    CallableJudge,
    Criterion,
    Evaluator,
    PromptJudge,
    Sample,
    load_jsonl,
    load_rubric,
    render_table,
)
from llm_eval_kit.text import content_tokens, recall

HERE = Path(__file__).parent


def answers_question(sample: Sample, criterion: Criterion) -> tuple[float, str]:
    """Rough relevance signal: how many of the question's content words the answer reuses."""
    question_terms = content_tokens(sample.question)
    overlap = recall(question_terms, content_tokens(sample.answer))
    return overlap, f"Answer reuses {overlap:.0%} of the question's key terms."


def offline_model(prompt: str) -> str:
    """Stand-in for a real model call. Swap this for your provider's SDK."""
    answer = prompt.split("ANSWER:", 1)[1]
    words = len(answer.split())
    score = 1.0 if 8 <= words <= 60 else 0.5
    return f"SCORE: {score}\nREASON: Answer length ({words} words) is appropriate for the question."


def main() -> None:
    rubric = load_rubric(HERE / "rubric.json")
    rubric.criteria += [
        Criterion(
            id="answers_question",
            description="The answer addresses the question that was asked.",
            judge="relevance",
            weight=1,
            severity="medium",
            threshold=0.5,
        ),
        Criterion(
            id="helpfulness",
            description="The answer is helpful and appropriately detailed.",
            judge="llm",
            weight=1,
            severity="low",
            threshold=0.7,
        ),
    ]
    judges = {
        "relevance": CallableJudge(answers_question),
        "llm": PromptJudge(complete=offline_model),
    }
    run = Evaluator(rubric, judges=judges).run(load_jsonl(HERE / "qa_dataset.jsonl"))
    print(render_table(run))


if __name__ == "__main__":
    main()
