"""I/O helpers: JSON read/write and GitHub REST API."""

import json
from pathlib import Path
from typing import Any

import httpx
import msgspec


def _to_serializable(data: Any) -> Any:
    """Convert msgspec Structs (and nested containers of them) to plain dicts."""
    if isinstance(data, msgspec.Struct):
        return msgspec.to_builtins(data)
    if isinstance(data, list):
        return [_to_serializable(item) for item in data]
    if isinstance(data, dict):
        return {k: _to_serializable(v) for k, v in data.items()}
    return data


def load_json(path: Path | str) -> Any:
    """Read and parse a JSON file as UTF-8.

    Parameters
    ----------
    path
        Filesystem path to a UTF-8 encoded JSON file.

    Returns
    -------
    Any
        The decoded JSON value — typically ``dict`` or ``list``,
        depending on the file's top-level shape.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path | str, data: Any, indent: int = 2, sort_keys: bool = False) -> None:
    """Write ``data`` as JSON to ``path``, creating parent directories as needed.

    Handles msgspec Structs transparently — they are converted to plain
    dicts before serialization.

    Parameters
    ----------
    path
        Destination filesystem path.
    data
        Any JSON-serializable value, including msgspec Structs.
    indent
        Number of spaces per indentation level (default 2 for diff-friendly output).
    sort_keys
        If True, sort dict keys alphabetically.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(_to_serializable(data), indent=indent, ensure_ascii=False, sort_keys=sort_keys),
        encoding="utf-8",
    )


def post_issue_comment(repo: str, issue_number: int, body: str, token: str) -> None:
    """Post a markdown comment to a GitHub issue via the REST API.

    Parameters
    ----------
    repo
        ``owner/repo`` slug.
    issue_number
        Numeric issue id to comment on.
    body
        Markdown body of the comment.
    token
        GitHub token with ``issues: write`` scope on ``repo``.
    """
    r = httpx.post(
        f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"body": body},
        timeout=30.0,
    )
    r.raise_for_status()
