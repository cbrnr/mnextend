# CLAUDE.md

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

1. Run `uv run tools/release.py prepare X.Y.Z` (with the version to be released). This removes the `.dev0` suffix from the `version` field in `pyproject.toml`, dates the `## [UNRELEASED]` heading in `CHANGELOG.md` with the version and today's date, and runs `uv lock`.
2. Review the resulting changes, then commit and push them.
3. Create a new release on GitHub and use the version as the tag name (make sure to prepend the version with a `v`, e.g. `v0.7.0`). Use `uv run tools/release.py notes X.Y.Z` to print the CHANGELOG entries for the release notes.
4. A GitHub Action takes care of building and uploading wheels to PyPI.

This concludes the new release. Now prepare the source for the next planned release as follows:

1. Run `uv run tools/release.py bump X.Y.Z` (with the next planned version). This sets the `version` field to `X.Y.Z.dev0`, starts a fresh `## [UNRELEASED] · YYYY-MM-DD` section at the top of `CHANGELOG.md`, and runs `uv lock`.
2. Commit ("Prepare next dev version") and push.
