import json
import os
import runpy
import subprocess
import sys

import pytest

from llm_eval_kit import (
    Evaluator,
    load_jsonl,
    load_rubric,
    render_table,
    to_markdown,
    write_reports,
)
from llm_eval_kit.cli import main


@pytest.fixture
def run(examples_dir):
    rubric = load_rubric(examples_dir / "rubric.json")
    return Evaluator(rubric).run(load_jsonl(examples_dir / "qa_dataset.jsonl"), dataset="qa.jsonl")


def test_terminal_table(run):
    table = render_table(run)
    assert "Rubric: grounded-qa v1.0" in table
    assert "Passed 3/8 (37.5%)" in table
    assert "consistency*" in table  # blocking failures are starred


def test_markdown_report_contains_justifications(run):
    md = to_markdown(run)
    assert md.startswith("# Evaluation report: grounded-qa")
    assert "### gift-card-refund - FAIL" in md
    assert "Polarity conflict" in md
    assert "| consistency (blocking) |" in md


def test_write_reports(tmp_path, run):
    paths = write_reports(run, tmp_path / "out")
    assert [p.name for p in paths] == ["report.md", "report.json"]
    data = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert data["summary"]["passed"] == 3
    assert data["results"][0]["criteria"][0]["justification"]
    with pytest.raises(ValueError):
        write_reports(run, tmp_path, formats=["html"])


def test_cli_run_writes_reports(tmp_path, examples_dir, capsys):
    code = main(
        [
            "run",
            "--dataset",
            str(examples_dir / "qa_dataset.jsonl"),
            "--rubric",
            str(examples_dir / "rubric.json"),
            "--out",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Passed 3/8" in out and "Wrote" in out
    assert (tmp_path / "report.md").exists() and (tmp_path / "report.json").exists()


def test_cli_exit_codes(examples_dir, tmp_path, capsys):
    base = [
        "run",
        "-d",
        str(examples_dir / "qa_dataset.jsonl"),
        "-r",
        str(examples_dir / "rubric.json"),
        "-q",
    ]
    assert main([*base, "--strict"]) == 1
    assert main([*base, "--fail-under", "0.3"]) == 0
    assert main([*base, "--fail-under", "0.5"]) == 1
    assert main([*base, "--pass-threshold", "0.0", "--fail-under", "0.3"]) == 0
    assert capsys.readouterr().out == ""

    assert (
        main(["run", "-d", str(tmp_path / "nope.jsonl"), "-r", str(examples_dir / "rubric.json")])
        == 2
    )
    assert "dataset not found" in capsys.readouterr().err


def test_cli_checks_and_validate(examples_dir, capsys):
    assert main(["checks"]) == 0
    assert "hallucination" in capsys.readouterr().out
    assert main(["validate", str(examples_dir / "rubric.json")]) == 0
    assert "OK: grounded-qa v1.0 - 5 criteria" in capsys.readouterr().out


def test_python_dash_m_entry_point(examples_dir):
    proc = subprocess.run(
        [sys.executable, "-m", "llm_eval_kit", "--version"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(examples_dir.parent / "src")},
    )
    assert proc.returncode == 0
    assert proc.stdout.strip().startswith("llm-eval 0.1.0")


def test_custom_judge_example_runs(examples_dir, capsys):
    runpy.run_path(str(examples_dir / "custom_judge.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "answers_question" in out and "helpfulness" in out
