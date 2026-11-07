# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-08

### Added

- Rubric model with weighted criteria, per-criterion thresholds and severities;
  loaders for JSON (always) and YAML (optional `yaml` extra).
- Deterministic checks: `hallucination`, `omission`, `contradiction`,
  `json_valid`, `length`, `required_keywords`, `forbidden_phrases`.
- Pluggable judge interface: `BaseJudge`, `CallableJudge` and a model-agnostic
  `PromptJudge` for LLM-as-judge setups.
- Batch evaluation of JSONL datasets with terminal, Markdown and JSON reports.
- `llm-eval` CLI with `run`, `checks` and `validate` commands, plus
  `--strict` / `--fail-under` exit codes for CI gating.
- Example dataset, rubrics and a custom-judge example.
