"""Shared helpers: paths, JSON I/O, sanitization, venue matching, rate limiting."""
from __future__ import annotations

import json
import re
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "state" / "features"
MODELS_DIR = ROOT / "models"
STATE_DIR = ROOT / "state"

SETLISTS_PATH = DATA_DIR / "setlists.json"
SONGS_PATH = DATA_DIR / "songs.json"
VENUES_PATH = DATA_DIR / "venues.json"
CANONICAL_NAMES_PATH = ROOT / "data" / "canonical_names.json"

VALID_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VALID_YEAR = re.compile(r"^\d{4}$")


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path | str, data: Any, indent: int = 2, sort_keys: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=sort_keys),
        encoding="utf-8",
    )


def sanitize(raw: str, max_length: int = 500) -> str:
    cleaned = raw.strip()[:max_length]
    return re.sub(r"[;&|`$(){}]", "", cleaned)


def parse_issue_form(body: str) -> dict[str, str]:
    """Parse GitHub Issue Form body into {label_lower: first_line_value}."""
    fields: dict[str, str] = {}
    sections = re.split(r"\n###\s+", "\n" + (body or ""))
    for section in sections[1:]:
        head, _, rest = section.partition("\n")
        value = rest.strip()
        if not value or value == "_No response_":
            continue
        fields[head.strip().lower()] = value.split("\n", 1)[0].strip()
    return fields


def normalize_venue_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    aliases = {
        "madison_square_garden": "msg",
        "the_garden": "msg",
        "saratoga_performing_arts_center": "spac",
    }
    return aliases.get(s, s)


def venue_id(name: str) -> str:
    return "v_" + normalize_venue_name(name)


def fuzzy_venue_match(target: str, venues: dict[str, dict]) -> str | None:
    if not target:
        return None
    direct = venue_id(target)
    if direct in venues:
        return direct
    target_norm = normalize_venue_name(target)
    best_score, best_id = 0.0, None
    for vid, v in venues.items():
        cand = normalize_venue_name(v.get("name", ""))
        if not cand:
            continue
        score = SequenceMatcher(None, target_norm, cand).ratio()
        if score > best_score:
            best_score, best_id = score, vid
    return best_id if best_score >= 0.7 else None


def check_rate_limit(
    usage: dict, user: str, max_per_day: int = 20, max_per_user: int = 3
) -> tuple[bool, str | None]:
    today = date.today().isoformat()
    if usage.get("date") != today:
        return True, None
    if usage.get("total", 0) >= max_per_day:
        return False, f"Daily limit of {max_per_day} predictions reached"
    if usage.get("by_user", {}).get(user, 0) >= max_per_user:
        return False, f"Per-user limit of {max_per_user} reached for @{user}"
    return True, None


def update_usage(usage: dict, user: str) -> dict:
    today = date.today().isoformat()
    if usage.get("date") != today:
        usage = {"date": today, "total": 0, "by_user": {}}
    usage["total"] = usage.get("total", 0) + 1
    usage["by_user"][user] = usage["by_user"].get(user, 0) + 1
    return usage


def day_of_week(date_str: str) -> str:
    return date.fromisoformat(date_str).strftime("%A").lower()


FESTIVAL_KEYWORDS = ("festival", "it ", "magnaball", "curveball", "dick's")


def special_show_flags(date_str: str, tour_name: str = "") -> dict[str, bool]:
    _, m, d = date_str.split("-")
    tour_lower = (tour_name or "").lower()
    return {
        "is_nye": (m, d) == ("12", "31"),
        "is_halloween": (m, d) == ("10", "31"),
        "is_festival": any(k in tour_lower for k in FESTIVAL_KEYWORDS),
    }


def canonicalize_song(name: str, canonical_map: dict[str, str] | None = None) -> str:
    if not name:
        return ""
    raw = name.strip()
    if canonical_map and raw in canonical_map:
        return canonical_map[raw]
    if canonical_map and raw.lower() in canonical_map:
        return canonical_map[raw.lower()]
    return raw
