# AGENTS.md

Guidelines for AI coding agents working on this repository.

## Project setup

- This project uses [uv](https://docs.astral.sh/uv/) for package and environment management.
- Install dependencies with `uv sync`.
- Run tests with `uv run pytest -W error tests` (CI turns warnings into errors, so match this locally to catch the same failures).

## Code style

- Formatting and linting are enforced by [Ruff](https://docs.astral.sh/ruff/). Run both of the following before committing:
  ```
  uv run ruff check --fix
  uv run ruff format
  ```
  The first command auto-fixes everything Ruff considers safe across the full lint rule set (`C4`, `E`, `F`, `I`, `PERF`, `UP`, `W` from `pyproject.toml`) and reports anything it can't fix automatically; the second formats the code. Together these match what CI checks (`ruff check` and `ruff format --diff`).
- Line length is 88 characters (the default). This limit applies to all code, including docstrings.
- Docstrings follow [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html), but use standard Markdown syntax instead of reStructuredText and a line length of 88 characters. In particular, inline code formatting uses single backticks (`` `x` ``), not double backticks (` ``x`` `) like in reStructuredText.
- Inline comments should start with a lower-case letter and be a single sentence where possible.

## Changelog

Every PR must include an entry in the `[UNRELEASED]` section of [CHANGELOG.md](CHANGELOG.md). Add it under the appropriate subsection (`### ✨ Added`, `### 🔧 Fixed`, `### 🌀 Changed`, or `### 🗑️ Removed`). Follow the existing style: a single sentence starting with a capital letter, followed by the PR link and author in parentheses, e.g.:

```
- Add support for XYZ ([#123](https://github.com/cbrnr/mnextend/pull/123) by [Your Name](https://github.com/yourname))
```

## Commit messages

- Use the imperative mood and start with a capital letter (e.g., `Fix crash when loading XDF files`).
- Keep the subject line concise (72 characters or fewer).

## Release

1. Remove the `.dev0` suffix from the `version` field in `pyproject.toml` (and/or adapt the version to be released if necessary).
2. Update the section in `CHANGELOG.md` corresponding to the new release with the version and current date.
3. Commit these changes and push.
4. Create a new release on GitHub and use the version as the tag name (make sure to prepend the version with a `v`, e.g. `v0.7.0`).
5. A GitHub Action takes care of building and uploading wheels to PyPI.

This concludes the new release. Now prepare the source for the next planned release as follows:

1. Update the `version` field in `pyproject.toml` to the next planned release and append `.dev0`.
2. Start a new section at the top of `CHANGELOG.md` titled `## [UNRELEASED] · YYYY-MM-DD`.
3. Commit ("Prepare next dev version") and push.
