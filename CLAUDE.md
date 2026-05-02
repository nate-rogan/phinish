# Phinish — Working Notes for Claude

## Skills to invoke

When editing this codebase, invoke these skills before writing code (they auto-trigger on description match, but reinforce here):

- **python-modern** — any `.py` file. Covers idioms, function decomposition, cross-module reuse.
- **pytest-modern** — any file under `tests/`. Use parametrize for multi-case tests; use `tmp_path` and `monkeypatch` instead of manual setup.
- **simplify** — after a meaningful change, run `/simplify` to catch reuse, quality, and efficiency issues across the diff.

## Project rules

- **Run via pixi**: `pixi run -e dev pytest`, `pixi run -e dev ruff check .`. Never invoke a bare `python` or `pytest` — they pick up the wrong env.
- **Package layout**: `src/phinish/` is the importable package, installed editable via `[tool.pixi.pypi-dependencies]`. Modules import from each other as `from phinish.X import Y`; no `sys.path` bootstrapping. CLI surface is exposed through `[project.scripts]` (`phinish-predict`, `phinish-scrape`, `phinish-process-issue`, etc.). For modules with argparse (`scrape`, `predict`), the entry point is `cli()` which calls `main(...)`; for the rest the entry point is `main()` directly.
- **Shared helpers go in `src/phinish/utils.py`** — see `show_song_set`, `min_max_normalize`, `post_issue_comment`, `SET_KEYS`, `SET_TO_INT`, `SET_DISPLAY`. Don't re-implement these per module.
- **Verify before claiming done**: run `pixi run -e dev ruff check . && pixi run -e dev pytest` and confirm both pass.

## Docstring policy

Use **NumPy-style** docstrings (enforced by ruff `D` rules with `convention = "numpy"`).

- **Module**: one-line `"""..."""` at the top describing what the module does. Required for every script.
- **Public functions** (no leading underscore): one-line summary. Required.
- **Non-trivial functions** (>30 lines, multi-step, or non-obvious behavior): full NumPy-style docstring with `Parameters`, `Returns`, and `Raises` sections where applicable. Encouraged.
- **Private helpers** (`_foo`): docstring optional; skip if the name + signature is self-explanatory.
- **Inline comments** stay rare — WHY only, never WHAT (the existing system-prompt rule).

**NumPy-style example:**

```python
def featurize(state: StreamingState, show: dict, song: str, cover_set: set[str]) -> list[float]:
    """Build the XGBoost feature vector for one (state, show, song) triple.

    Parameters
    ----------
    state
        Streaming statistics accumulated from all shows strictly before `show`.
    show
        The target show being scored; only its date/venue/tour fields are read.
    song
        Candidate song name.
    cover_set
        Set of song names considered covers (non-Phish originals).

    Returns
    -------
    list[float]
        A 31-element feature vector aligned with `feature_names()`.
    """
```

NumPy convention rules (the ones ruff enforces):
- Summary on the first line (D200, D205).
- Section headers (`Parameters`, `Returns`, `Raises`, etc.) underlined with `----`.
- Parameter names without type after them — types come from the signature.
- Blank line between summary and the first section.

## Function size

- `main()` should orchestrate, not implement. If `main()` has 5+ phases, extract each as a `_named_phase()` helper.
- Don't decompose single conceptual operations (one feature vector, one streaming accumulator) — splitting by category just scatters context.

## Testing

- Hand-verified expected values for feature engineering live in `tests/conftest.py::tiny_shows`.
- New helpers in `src/phinish/utils.py` should get a unit test before they get a second caller.
- For the predict pipeline, prefer testing the public API (`predict()`, `format_comment()`) over internals.
