# Contributing

Thanks for your interest in improving llm-eval-kit. Bug reports, new checks and
documentation fixes are all welcome.

## Development setup

```bash
git clone https://github.com/s78519300-creator/llm-eval-kit.git
cd llm-eval-kit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check .
ruff format --check .
pytest
```

CI runs the same commands on Python 3.9 through 3.12.

## Guidelines

- **No runtime dependencies.** The core package is pure standard library; optional
  integrations belong behind an extra (as with `yaml`).
- **Checks must be deterministic** and explain themselves: every `CheckResult`
  needs a justification a reviewer can act on, and evidence where it helps.
- **Add tests** for new behaviour, including at least one failing case.
- Keep pull requests focused, and add a line to `CHANGELOG.md` under an
  "Unreleased" heading.

## Adding a new check

1. Write a function `my_check(sample, **params) -> CheckResult` in
   `src/llm_eval_kit/checks.py` and decorate it with `@register_check("my_check")`.
2. Give it a one-line docstring summary (it is shown by `llm-eval checks`).
3. Add tests in `tests/test_checks.py` and document it in the README.
