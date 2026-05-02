"""Pure-ish helpers: sanitization, venue matching, dates, rate limiting, show utilities."""

import re
from datetime import date
from difflib import SequenceMatcher
from typing import NamedTuple

from phinish.process.types import Usage
from phinish.scrape.types import Show, VenueRecord
from phinish.utils.constants import FUZZY_VENUE_THRESHOLD

# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------


def sanitize(raw: str, max_length: int = 500) -> str:
    r"""Trim, length-cap, and strip shell-metachars from untrusted user input.

    Parameters
    ----------
    raw
        Raw user-supplied string (issue body field, env var, etc.).
    max_length
        Maximum length to keep before stripping metacharacters.

    Returns
    -------
    str
        The cleaned string, safe to embed in markdown / log lines.
    """
    cleaned = raw.strip()[:max_length]
    return re.sub(r"[;&|`$(){}]", "", cleaned)


def parse_issue_form(body: str) -> dict[str, str]:
    """Parse a GitHub Issue Form body into a flat ``{label: value}`` mapping.

    Parameters
    ----------
    body
        Raw markdown body of a labeled issue. Empty string is allowed.

    Returns
    -------
    dict[str, str]
        Mapping of lowercased label to the first line of its value.
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


def canonicalize_song(name: str, canonical_map: dict[str, str] | None = None) -> str:
    """Resolve a raw song name to its canonical form via the alias map.

    Parameters
    ----------
    name
        Raw song name from a setlist row.
    canonical_map
        Optional ``{variant -> canonical}`` mapping.

    Returns
    -------
    str
        Canonical name if a mapping exists, otherwise the trimmed input.
    """
    if not name:
        return ""
    raw = name.strip()
    if canonical_map and raw in canonical_map:
        return canonical_map[raw]
    if canonical_map and raw.lower() in canonical_map:
        return canonical_map[raw.lower()]
    return raw


# ---------------------------------------------------------------------------
# Venue resolution
# ---------------------------------------------------------------------------


def normalize_venue_name(name: str) -> str:
    """Lowercase, snake-case, and resolve common aliases (MSG, SPAC, ...).

    Parameters
    ----------
    name
        Raw venue name as it appears in API data or user input.

    Returns
    -------
    str
        Normalized slug — the body of a canonical ``venue_id``.
    """
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    aliases = {
        "madison_square_garden": "msg",
        "the_garden": "msg",
        "saratoga_performing_arts_center": "spac",
    }
    return aliases.get(s, s)


def venue_id(name: str) -> str:
    """Return the canonical ``v_<slug>`` venue identifier for a venue name.

    Parameters
    ----------
    name
        Raw venue name.

    Returns
    -------
    str
        The ``v_``-prefixed canonical identifier.
    """
    return "v_" + normalize_venue_name(name)


def fuzzy_venue_match(target: str, venues: dict[str, VenueRecord]) -> str | None:
    """Resolve a venue name (possibly mistyped) to a known venue id.

    Parameters
    ----------
    target
        Venue name as the user typed it. Empty input returns ``None``.
    venues
        Catalog from ``data/processed/venues.json``, keyed by canonical
        venue id.

    Returns
    -------
    str or None
        The matched canonical ``venue_id``, or ``None``.
    """
    if not target:
        return None
    direct = venue_id(target)
    if direct in venues:
        return direct
    target_norm = normalize_venue_name(target)
    best_score, best_id = 0.0, None
    for vid, v in venues.items():
        cand = normalize_venue_name(v.name)
        if not cand:
            continue
        score = SequenceMatcher(None, target_norm, cand).ratio()
        if score > best_score:
            best_score, best_id = score, vid
    return best_id if best_score >= FUZZY_VENUE_THRESHOLD else None


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

FESTIVAL_KEYWORDS = ("festival", "it ", "magnaball", "curveball", "dick's")


class ShowFlags(NamedTuple):
    """Boolean flags for special show types."""

    is_nye: bool
    is_halloween: bool
    is_festival: bool


def day_of_week(d: str | date) -> str:
    """Return the lowercase day name (``'monday'``, ...) for a date.

    Parameters
    ----------
    d
        A ``date`` object or an ISO date string (``YYYY-MM-DD``).

    Returns
    -------
    str
        Lowercase English weekday name.
    """
    if isinstance(d, str):
        d = date.fromisoformat(d)
    return d.strftime("%A").lower()


def special_show_flags(date_str: str, tour_name: str = "") -> ShowFlags:
    """Flag NYE, Halloween, and festival shows by date and tour name.

    Parameters
    ----------
    date_str
        ISO date as ``YYYY-MM-DD``.
    tour_name
        Tour / festival name from the show metadata; ``''`` if unknown.

    Returns
    -------
    ShowFlags
        Typed triple of boolean flags.
    """
    _, m, d = date_str.split("-")
    tour_lower = (tour_name or "").lower()
    return ShowFlags(
        is_nye=(m, d) == ("12", "31"),
        is_halloween=(m, d) == ("10", "31"),
        is_festival=any(k in tour_lower for k in FESTIVAL_KEYWORDS),
    )


# ---------------------------------------------------------------------------
# Show utilities
# ---------------------------------------------------------------------------


def show_song_set(show: Show) -> set[str]:
    """Return the unique set of songs played across all sets of one show.

    Parameters
    ----------
    show
        A ``Show`` struct with one or more sets.

    Returns
    -------
    set[str]
        Distinct song names.
    """
    return {s.song for songs in show.sets.values() for s in songs}


def min_max_normalize(values: list[float]) -> list[float]:
    """Rescale values to ``[0, 1]``; constant or empty input maps to zeros.

    Parameters
    ----------
    values
        Numeric values to rescale.

    Returns
    -------
    list[float]
        Same length as input, each value rescaled to ``[0, 1]``.
    """
    if not values:
        return values
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    return [(v - lo) / rng for v in values]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def check_rate_limit(
    usage: Usage, user: str, max_per_day: int = 20, max_per_user: int = 3
) -> tuple[bool, str | None]:
    """Return ``(ok, reason)`` for a prediction request given the day's usage.

    Parameters
    ----------
    usage
        The persisted ``Usage`` counter.
    user
        GitHub login of the requesting user.
    max_per_day
        Maximum predictions allowed per UTC day across all users.
    max_per_user
        Maximum predictions allowed per UTC day for a single user.

    Returns
    -------
    tuple[bool, str or None]
        ``(True, None)`` if the request is allowed; ``(False, reason)``
        with a human-readable explanation if blocked.
    """
    today = date.today().isoformat()
    if usage.date != today:
        return True, None
    if usage.total >= max_per_day:
        return False, f"Daily limit of {max_per_day} predictions reached"
    if usage.by_user.get(user, 0) >= max_per_user:
        return False, f"Per-user limit of {max_per_user} reached for @{user}"
    return True, None


def update_usage(usage: Usage, user: str) -> Usage:
    """Increment today's prediction counters, resetting on a date rollover.

    Parameters
    ----------
    usage
        Existing usage counter, possibly from a prior day.
    user
        GitHub login of the user whose request just succeeded.

    Returns
    -------
    Usage
        The updated counter.
    """
    today = date.today().isoformat()
    if usage.date != today:
        return Usage(date=today, total=1, by_user={user: 1})
    new_by_user = dict(usage.by_user)
    new_by_user[user] = new_by_user.get(user, 0) + 1
    return Usage(date=today, total=usage.total + 1, by_user=new_by_user)
