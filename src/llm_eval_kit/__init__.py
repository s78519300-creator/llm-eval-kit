"""llm-eval-kit: lightweight, rubric-based evaluation of LLM outputs."""

from __future__ import annotations

from .checks import CheckResult, available_checks, get_check, register_check
from .dataset import DatasetError, Sample, load_jsonl
from .evaluator import CriterionResult, Evaluator, RunResult, SampleResult
from .judges import BaseJudge, CallableJudge, CheckJudge, PromptJudge, parse_judgement
from .report import render_table, to_json, to_markdown, write_reports
from .rubric import Criterion, Rubric, RubricError, load_rubric

__version__ = "0.1.0"

__all__ = [
    "BaseJudge",
    "CallableJudge",
    "CheckJudge",
    "CheckResult",
    "Criterion",
    "CriterionResult",
    "DatasetError",
    "Evaluator",
    "PromptJudge",
    "Rubric",
    "RubricError",
    "RunResult",
    "Sample",
    "SampleResult",
    "__version__",
    "available_checks",
    "get_check",
    "load_jsonl",
    "load_rubric",
    "parse_judgement",
    "register_check",
    "render_table",
    "to_json",
    "to_markdown",
    "write_reports",
]
