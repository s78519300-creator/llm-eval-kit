"""Terminal, Markdown and JSON renderings of an evaluation run."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Union

from .evaluator import RunResult, SampleResult
from .text import shorten


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _failed_label(result: SampleResult) -> str:
    if not result.failed:
        return "-"
    labels = [c.criterion_id + ("*" if c.blocking else "") for c in result.failed]
    return ", ".join(labels)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], right: Sequence[int] = ()) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    def fmt(cells: Sequence[str]) -> str:
        parts = [
            cell.rjust(w) if i in right else cell.ljust(w)
            for i, (cell, w) in enumerate(zip(cells, widths))
        ]
        return "  ".join(parts).rstrip()

    rule = "  ".join("-" * w for w in widths)
    return "\n".join([fmt(headers), rule, *(fmt(r) for r in rows)])


def render_table(run: RunResult) -> str:
    """Plain-text summary suitable for a terminal or CI log."""
    rubric = run.rubric
    header = f"Rubric: {rubric.name} v{rubric.version}  |  samples: {len(run.results)}"
    if run.dataset:
        header += f"  |  dataset: {run.dataset}"

    sample_rows = [
        [r.sample_id, f"{r.score:.2f}", _status(r.passed), _failed_label(r)] for r in run.results
    ]
    samples = _table(["SAMPLE", "SCORE", "RESULT", "FAILED CRITERIA"], sample_rows, right=[1])

    crit_rows = [
        [
            cid,
            f"{s['weight']:g}",
            s["severity"],
            f"{s['mean_score']:.2f}",
            f"{s['passed']}/{s['total']}",
        ]
        for cid, s in run.criterion_stats().items()
    ]
    criteria = _table(
        ["CRITERION", "WEIGHT", "SEVERITY", "MEAN", "PASSED"], crit_rows, right=[1, 3, 4]
    )

    totals = (
        f"Passed {run.passed}/{len(run.results)} ({run.pass_rate:.1%})  |  "
        f"mean score {run.mean_score:.2f}  |  pass threshold {rubric.pass_threshold:.2f}"
    )
    legend = "* = blocking criterion (severity: " + ", ".join(rubric.blocking_severities) + ")"
    return "\n".join([header, "", samples, "", criteria, "", totals, legend])


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(run: RunResult, max_evidence: int = 5) -> str:
    """Full, auditable Markdown report with per-criterion justifications."""
    rubric = run.rubric
    lines = [
        f"# Evaluation report: {rubric.name}",
        "",
        f"- **Rubric:** {rubric.name} v{rubric.version}",
        f"- **Dataset:** {run.dataset or 'n/a'}",
        f"- **Run started:** {run.started_at}",
        f"- **Pass threshold:** {rubric.pass_threshold:.2f} "
        f"(blocking severities: {', '.join(rubric.blocking_severities)})",
        "",
        "## Summary",
        "",
        "| Samples | Passed | Failed | Pass rate | Mean score |",
        "|---:|---:|---:|---:|---:|",
        f"| {len(run.results)} | {run.passed} | {len(run.results) - run.passed} "
        f"| {run.pass_rate:.1%} | {run.mean_score:.2f} |",
        "",
        "### By criterion",
        "",
        "| Criterion | Scorer | Weight | Severity | Threshold | Mean score | Passed |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    stats = run.criterion_stats()
    for c in rubric.criteria:
        s = stats[c.id]
        lines.append(
            f"| {c.id} | `{c.scorer}` | {c.weight:g} | {c.severity} | {c.threshold:.2f} "
            f"| {s['mean_score']:.2f} | {s['passed']}/{s['total']} |"
        )

    lines += [
        "",
        "### By sample",
        "",
        "| Sample | Score | Result | Failed criteria |",
        "|---|---:|---|---|",
    ]
    for r in run.results:
        lines.append(
            f"| {r.sample_id} | {r.score:.2f} | {_status(r.passed)} | {_failed_label(r)} |"
        )

    lines += ["", "## Details", ""]
    for r in run.results:
        lines += [f"### {r.sample_id} - {_status(r.passed)} ({r.score:.2f})", ""]
        if r.question:
            lines += [f"**Question:** {r.question}", ""]
        lines += [f"**Answer:** {shorten(r.answer, 400)}", "", f"**Verdict:** {r.verdict}", ""]
        lines += [
            "| Criterion | Score | Threshold | Result | Justification |",
            "|---|---:|---:|---|---|",
        ]
        for c in r.criteria:
            label = c.criterion_id + (" (blocking)" if c.blocking else "")
            lines.append(
                f"| {label} | {c.score:.2f} | {c.threshold:.2f} | {_status(c.passed)} "
                f"| {_md_escape(c.justification)} |"
            )
        evidence = [(c.criterion_id, e) for c in r.criteria for e in c.evidence]
        if evidence:
            lines += ["", "Evidence:", ""]
            for cid, item in evidence[:max_evidence]:
                lines.append(f"- `{cid}`: {item}")
            if len(evidence) > max_evidence:
                lines.append(f"- ...and {len(evidence) - max_evidence} more (see JSON report)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def to_json(run: RunResult) -> str:
    """Machine-readable report containing every score, justification and piece of evidence."""
    return json.dumps(run.to_dict(), indent=2, ensure_ascii=False) + "\n"


def write_reports(
    run: RunResult,
    out_dir: Union[str, Path],
    formats: Sequence[str] = ("md", "json"),
    stem: str = "report",
) -> list[Path]:
    """Write the requested report formats into ``out_dir`` and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        if fmt == "md":
            path = out / f"{stem}.md"
            path.write_text(to_markdown(run), encoding="utf-8")
        elif fmt == "json":
            path = out / f"{stem}.json"
            path.write_text(to_json(run), encoding="utf-8")
        else:
            raise ValueError(f"unknown report format {fmt!r} (use 'md' or 'json')")
        written.append(path)
    return written
