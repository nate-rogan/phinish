"""Shared helpers: paths, JSON I/O, sanitization, venue matching, rate limiting."""
from __future__ import annotations

import json
import re
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "state" / "features"
MODELS_DIR = ROOT / "models"
STATE_DIR = ROOT / "state"

SETLISTS_PATH = DATA_DIR / "setlists.json"
SONGS_PATH = DATA_DIR / "songs.json"
VENUES_PATH = DATA_DIR / "venues.json"
CANONICAL_NAMES_PATH = ROOT / "data" / "canonical_names.json"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
STATE_SNAPSHOT_PATH = MODELS_DIR / "state.pkl"

VALID_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VALID_YEAR = re.compile(r"^\d{4}$")

SET_KEYS: tuple[str, ...] = ("1", "2", "3", "encore")
SET_DISPLAY: tuple[tuple[str, str], ...] = (("1", "Set 1"), ("2", "Set 2"), ("encore", "Encore"))
SET_TO_INT: dict[str, int] = {"1": 1, "2": 2, "3": 3, "encore": 3}
TOP_K: int = 25
FUZZY_VENUE_THRESHOLD: float = 0.7


# --- Setlist domain types (data/processed/) ---


class SongEntry(TypedDict):
    """One song slot inside a set: position, transition mark, jam/reprise flags."""

    song: str
    song_id: str
    position: int
    transition: str
    is_jam: bool
    is_reprise: bool


class Show(TypedDict):
    """One Phish show with date, venue, tour metadata, and all sets played.

    Persisted as one entry in `data/processed/setlists.json`. Also produced
    in-memory by `predict.synthetic_show` for inference (without `total_songs`).
    """

    show_id: str
    date: str
    year: int
    month: int
    day: int
    day_of_week: str
    venue_id: str
    venue_name: str
    city: str
    state: str
    country: str
    tour: str
    tour_id: str
    sets: dict[str, list[SongEntry]]
    is_nye: bool
    is_halloween: bool
    is_festival: bool
    total_songs: NotRequired[int]


class SongCatalogEntry(TypedDict):
    """One row from the Phish.net song catalog (`data/processed/songs.json`)."""

    song_id: str
    name: str
    slug: str
    artist: str
    is_original: bool
    debut: str
    last_played: str
    times_played: int


class VenueRecord(TypedDict):
    """One venue from `data/processed/venues.json`, keyed by `venue_id`."""

    venue_id: str
    name: str
    city: str
    state: str
    country: str
    phishnet_id: NotRequired[str]


# --- Feature store types (state/features/) ---


class SongStats(TypedDict):
    """Per-song statistics (`state/features/song_stats.json`)."""

    total_plays: int
    lifetime_frequency: float
    recent_frequency_50: float
    recent_frequency_20: float
    avg_set_position: float
    typical_set: int
    set_distribution: dict[str, float]
    opener_frequency: float
    closer_frequency: float
    is_cover: bool
    debut_year: int


class SongGap(TypedDict):
    """Per-song gap snapshot (`state/features/song_gaps.json`)."""

    gap: int
    last_played: str


class VenueHistory(TypedDict):
    """Per-venue history (`state/features/venue_history.json`)."""

    venue_id: str
    name: str
    total_shows: int
    song_freq: dict[str, float]
    common_openers: list[str]
    common_closers: list[str]


# Mapping of song -> probability used inside transition matrices.
TransitionDist = dict[str, float]


class TransitionMatrix(TypedDict):
    """Order-1 / order-2 transitions plus opener / closer marginals.

    Persisted to `state/features/transition_matrix.json` (no
    `trained_on_shows`) and to `models/markov_order2.json` (with
    `trained_on_shows`).
    """

    order_1: dict[str, TransitionDist]
    order_2: dict[str, TransitionDist]
    set_openers: dict[str, TransitionDist]
    set_closers: dict[str, TransitionDist]
    vocab_size: int
    laplace_k: float
    trained_on_shows: NotRequired[int]


# --- Model artifacts (models/) ---


class EnsembleWeights(TypedDict):
    """Optimal ensemble weights (`models/ensemble_weights.json`)."""

    w_xgboost: float
    w_markov: float
    w_gap: float
    w_venue: float
    val_precision_at_25: float | None
    val_year: int
    n_val_shows: int


# --- Prediction output ---


class PredictionItem(TypedDict):
    """One song slot in a prediction setlist with calibrated confidence."""

    song: str
    confidence: float
    gap: int


class PredictionWeights(TypedDict):
    """Ensemble weights echoed back in the prediction payload."""

    xgboost: float
    markov: float
    gap: float
    venue: float


class Prediction(TypedDict):
    """Output of `predict.predict()` — structured 3-set forecast."""

    date: str
    venue: str
    venue_id: str
    city: str
    setlist: dict[str, list[PredictionItem]]
    avg_confidence: float
    model_version: int | str
    weights: PredictionWeights


# --- Mutable run-state ---


class Usage(TypedDict, total=False):
    """Daily prediction-counter (`state/usage.json`); empty on a fresh day."""

    date: str
    total: int
    by_user: dict[str, int]


def load_json(path: Path | str) -> Any:
    """Read and parse a JSON file as UTF-8."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path | str, data: Any, indent: int = 2, sort_keys: bool = False) -> None:
    """Write `data` as JSON to `path`, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=sort_keys),
        encoding="utf-8",
    )


def sanitize(raw: str, max_length: int = 500) -> str:
    """Trim, length-cap, and strip shell-metachars from untrusted user input."""
    cleaned = raw.strip()[:max_length]
    return re.sub(r"[;&|`$(){}]", "", cleaned)


def parse_issue_form(body: str) -> dict[str, str]:
    """Parse a GitHub Issue Form body into a flat ``{label: value}`` mapping.

    GitHub renders form submissions as a sequence of ``### Label`` blocks,
    each followed by a blank line and the user's value (or the literal
    ``_No response_`` for empty optional fields). This walks those blocks,
    lowercases each label, and keeps only the first non-empty line of the
    value — multi-line answers and unanswered fields are dropped.

    Parameters
    ----------
    body
        Raw markdown body of a labeled issue (typically from
        ``github.event.issue.body``). Empty string is allowed.

    Returns
    -------
    dict[str, str]
        Mapping of lowercased label to the first line of its value. Skipped
        labels (empty or ``_No response_``) are absent rather than mapped to
        an empty string, so callers can use ``fields.get("year", "")`` with
        a meaningful default.
    """
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
    """Lowercase, snake-case, and resolve common aliases (MSG, SPAC, etc)."""
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    aliases = {
        "madison_square_garden": "msg",
        "the_garden": "msg",
        "saratoga_performing_arts_center": "spac",
    }
    return aliases.get(s, s)


def venue_id(name: str) -> str:
    """Return the canonical `v_<slug>` venue identifier for a venue name."""
    return "v_" + normalize_venue_name(name)


def fuzzy_venue_match(target: str, venues: dict[str, VenueRecord]) -> str | None:
    """Resolve a venue name (possibly mistyped) to a known venue id.

    Tries the alias-resolved canonical id first (``v_<slug>``); if absent,
    falls back to a similarity scan of every venue name in ``venues`` using
    ``difflib.SequenceMatcher`` and returns the best match above
    ``FUZZY_VENUE_THRESHOLD``.

    Parameters
    ----------
    target
        Venue name as the user typed it. Empty input returns ``None``.
    venues
        Catalog from ``data/processed/venues.json``, keyed by canonical
        venue id. Entries without a ``name`` are skipped.

    Returns
    -------
    str or None
        The matched canonical ``venue_id`` if either the direct lookup hits
        or the best fuzzy match meets the similarity threshold; otherwise
        ``None`` so the caller can decide how to handle the unknown venue.
    """
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
    return best_id if best_score >= FUZZY_VENUE_THRESHOLD else None


def check_rate_limit(
    usage: Usage, user: str, max_per_day: int = 20, max_per_user: int = 3
) -> tuple[bool, str | None]:
    """Return (ok, reason) for a prediction request given the day's usage."""
    today = date.today().isoformat()
    if usage.get("date") != today:
        return True, None
    if usage.get("total", 0) >= max_per_day:
        return False, f"Daily limit of {max_per_day} predictions reached"
    if usage.get("by_user", {}).get(user, 0) >= max_per_user:
        return False, f"Per-user limit of {max_per_user} reached for @{user}"
    return True, None


def update_usage(usage: Usage, user: str) -> Usage:
    """Increment today's prediction counters, resetting on a date rollover."""
    today = date.today().isoformat()
    if usage.get("date") != today:
        usage = {"date": today, "total": 0, "by_user": {}}
    usage["total"] = usage.get("total", 0) + 1
    usage["by_user"][user] = usage["by_user"].get(user, 0) + 1
    return usage


def day_of_week(date_str: str) -> str:
    """Return the lowercase day name ('monday', ...) for an ISO date string."""
    return date.fromisoformat(date_str).strftime("%A").lower()


FESTIVAL_KEYWORDS = ("festival", "it ", "magnaball", "curveball", "dick's")


def special_show_flags(date_str: str, tour_name: str = "") -> dict[str, bool]:
    """Flag NYE, Halloween, and festival shows by date and tour name."""
    _, m, d = date_str.split("-")
    tour_lower = (tour_name or "").lower()
    return {
        "is_nye": (m, d) == ("12", "31"),
        "is_halloween": (m, d) == ("10", "31"),
        "is_festival": any(k in tour_lower for k in FESTIVAL_KEYWORDS),
    }


def canonicalize_song(name: str, canonical_map: dict[str, str] | None = None) -> str:
    """Resolve a raw song name to its canonical form via the alias map."""
    if not name:
        return ""
    raw = name.strip()
    if canonical_map and raw in canonical_map:
        return canonical_map[raw]
    if canonical_map and raw.lower() in canonical_map:
        return canonical_map[raw.lower()]
    return raw


def show_song_set(show: Show) -> set[str]:
    """Return the unique set of songs played across all sets of one show."""
    return {s["song"] for songs in show.get("sets", {}).values() for s in songs}


def min_max_normalize(values: list[float]) -> list[float]:
    """Rescale values to [0, 1]; constant or empty input maps to all zeros."""
    if not values:
        return values
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    return [(v - lo) / rng for v in values]


def post_issue_comment(repo: str, issue_number: int, body: str, token: str) -> None:
    """Post a markdown comment to a GitHub issue via the REST API."""
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
