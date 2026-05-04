"""Phish.net API v5 scraper.

Pulls setlists, songs, venues and writes:
  data/processed/setlists.json
  data/processed/songs.json
  data/processed/venues.json

Usage:
  pixi run python -m phinish.scrape.api              # full pull (1983-present)
  pixi run python -m phinish.scrape.api --year 2025  # one year, merged into existing

Requires PHISHNET_API_KEY environment variable.
"""

import argparse
import os
from collections import defaultdict
from datetime import date

import httpx
import msgspec
import stamina
import structlog

from phinish.scrape.types import (
    ApiSetlistRow,
    ApiSongRow,
    ApiVenueRow,
    Show,
    SongCatalogEntry,
    SongEntry,
    VenueRecord,
)
from phinish.utils import (
    CANONICAL_NAMES_PATH,
    DATA_DIR,
    SETLISTS_PATH,
    SONGS_PATH,
    VENUES_PATH,
    canonicalize_song,
    day_of_week,
    load_json,
    save_json,
    special_show_flags,
    venue_id,
)

log = structlog.get_logger()

API_BASE = "https://api.phish.net/v5"
FIRST_YEAR = 1983
TIMEOUT = 120.0


def _is_retryable(exc: BaseException) -> bool:
    """Retry on network errors, 5xx, and 429 — not on other 4xx (e.g. bad key)."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, (httpx.RequestError, RuntimeError))


def get_api_key() -> str:
    """Read ``PHISHNET_API_KEY`` from the environment or exit with a clear message.

    Returns
    -------
    str
        The API key value, never empty.

    Raises
    ------
    SystemExit
        If ``PHISHNET_API_KEY`` is unset or empty.
    """
    key = os.environ.get("PHISHNET_API_KEY")
    if not key:
        raise SystemExit("PHISHNET_API_KEY environment variable is required.")
    return key


@stamina.retry(on=_is_retryable, attempts=5)
def fetch(client: httpx.Client, path: str, key: str) -> list[dict]:
    """GET an API path and return its ``data`` payload.

    Parameters
    ----------
    client
        Reusable ``httpx.Client``.
    path
        Path component beneath ``/v5/``.
    key
        Phish.net API key.

    Returns
    -------
    list[dict]
        Raw rows from the API's ``data`` field.
    """
    url = f"{API_BASE}/{path.lstrip('/')}"
    r = client.get(url, params={"apikey": key}, timeout=TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(f"{path}: {payload.get('error_message')}")
    return payload.get("data", []) or []


def group_into_shows(rows: list[dict], canonical: dict[str, str]) -> list[Show]:
    """Collapse Phish.net per-song rows into per-show records sorted by date.

    Parameters
    ----------
    rows
        Raw rows as returned by ``GET /v5/setlists/showyear/<year>.json``.
    canonical
        Map of raw song name to canonical name for deduplication.

    Returns
    -------
    list[Show]
        One ``Show`` per unique ``showid``, sorted by date.
    """
    # First pass: accumulate sets as mutable dicts keyed by showid
    show_meta: dict[str, ApiSetlistRow] = {}
    sets_acc: dict[str, dict[str, list[SongEntry]]] = defaultdict(lambda: defaultdict(list))
    for raw in rows:
        # Phish.net API returns some fields as int inconsistently; stringify everything
        # except position (genuinely numeric) so msgspec can parse cleanly.
        normalized = {k: (v if k == "position" else str(v)) for k, v in raw.items()}
        row = msgspec.convert(normalized, ApiSetlistRow, strict=False)
        sid = row.showid
        if not sid:
            continue
        if sid not in show_meta:
            show_meta[sid] = row
        entry = SongEntry(
            song=canonicalize_song(row.song, canonical),
            song_id=row.songid,
            position=row.position,
            transition=row.transition_mark,
            is_jam=row.is_jam,
            is_reprise=row.is_reprise,
        )
        sets_acc[sid][row.set_key].append(entry)

    # Second pass: build immutable Show structs
    out: list[Show] = []
    for sid, first_row in show_meta.items():
        d = date.fromisoformat(first_row.showdate)
        sets = {
            k: sorted(v, key=lambda s: s.position)
            for k, v in sets_acc[sid].items() if v
        }
        flags = special_show_flags(d.isoformat(), first_row.tourname)
        out.append(Show(
            show_id=sid,
            date=d.isoformat(),
            year=d.year,
            month=d.month,
            day=d.day,
            day_of_week=day_of_week(d),
            venue_id=venue_id(first_row.venue),
            venue_name=first_row.venue,
            city=first_row.city,
            state=first_row.state,
            country=first_row.country,
            tour=first_row.tourname,
            tour_id=first_row.tourid,
            sets=sets,
            is_nye=flags.is_nye,
            is_halloween=flags.is_halloween,
            is_festival=flags.is_festival,
            total_songs=sum(len(v) for v in sets.values()),
        ))
    out.sort(key=lambda s: s.date)
    return out


def merge_setlists(existing: list[Show], new: list[Show]) -> list[Show]:
    """Merge new shows into existing, replacing duplicates by ``show_id``.

    Parameters
    ----------
    existing
        Previously persisted shows (full history).
    new
        Newly scraped shows for one year, possibly overlapping.

    Returns
    -------
    list[Show]
        Combined shows, deduplicated by ``show_id``, sorted by ``date``.
    """
    by_id = {s.show_id: s for s in existing}
    for s in new:
        by_id[s.show_id] = s
    return sorted(by_id.values(), key=lambda s: s.date)


def normalize_songs(rows: list[dict]) -> list[SongCatalogEntry]:
    """Reshape Phish.net song catalog rows into the project's schema.

    Parameters
    ----------
    rows
        Raw rows from ``GET /v5/songs.json``.

    Returns
    -------
    list[SongCatalogEntry]
        One typed entry per song.
    """
    out: list[SongCatalogEntry] = []
    for raw in rows:
        normalized = {k: (v if k == "times_played" else str(v)) for k, v in raw.items()}
        row = msgspec.convert(normalized, ApiSongRow, strict=False)
        artist = row.resolved_artist
        out.append(SongCatalogEntry(
            song_id=row.resolved_id,
            name=row.resolved_name,
            slug=row.slug,
            artist=artist,
            is_original=artist.lower() == "phish",
            debut=row.debut,
            last_played=row.last_played,
            times_played=row.times_played,
        ))
    return out


def normalize_venues(rows: list[dict]) -> dict[str, VenueRecord]:
    """Reshape Phish.net venue rows into ``{venue_id: VenueRecord}``.

    Parameters
    ----------
    rows
        Raw rows from ``GET /v5/venues.json``.

    Returns
    -------
    dict[str, VenueRecord]
        Mapping of canonical ``venue_id`` to a typed ``VenueRecord``.
    """
    venues: dict[str, VenueRecord] = {}
    for raw in rows:
        normalized = {k: str(v) for k, v in raw.items()}
        row = msgspec.convert(normalized, ApiVenueRow, strict=False)
        name = row.resolved_name
        if not name:
            continue
        vid = venue_id(name)
        venues[vid] = VenueRecord(
            venue_id=vid,
            phishnet_id=row.resolved_id,
            name=name,
            city=row.city,
            state=row.state,
            country=row.country,
        )
    return venues


def scrape(year: int | None = None) -> None:
    """Pull setlists/songs/venues from Phish.net and persist to data/source/.

    Parameters
    ----------
    year
        If provided, scrape only that year and merge into the existing
        ``setlists.json``. If ``None`` (default), do a full pull from
        ``FIRST_YEAR`` to the current year and overwrite ``setlists.json``.
    """
    key = get_api_key()
    canonical = load_json(CANONICAL_NAMES_PATH) if CANONICAL_NAMES_PATH.exists() else {}
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if year is not None and not SETLISTS_PATH.exists():
        log.warning("no_existing_setlists", year=year, action="falling back to full scrape")
        year = None

    with httpx.Client() as client:
        if year is None:
            years = range(FIRST_YEAR, date.today().year + 1)
            all_rows: list[dict] = []
            for y in years:
                log.info("fetching_setlists", year=y)
                all_rows.extend(fetch(client, f"setlists/showyear/{y}.json", key))
            shows = group_into_shows(all_rows, canonical)
            save_json(SETLISTS_PATH, shows)
            log.info("wrote_setlists", shows=len(shows), path=str(SETLISTS_PATH))
        else:
            log.info("fetching_setlists", year=year)
            rows = fetch(client, f"setlists/showyear/{year}.json", key)
            new_shows = group_into_shows(rows, canonical)
            existing_raw = load_json(SETLISTS_PATH) if SETLISTS_PATH.exists() else []
            existing = [msgspec.convert(s, Show, strict=False) for s in existing_raw]
            merged = merge_setlists(existing, new_shows)
            save_json(SETLISTS_PATH, merged)
            log.info("merged_setlists", year=year, new=len(new_shows), total=len(merged))

        # Catalog endpoints are large and slow; skip on incremental runs
        # when the files already exist.
        if year is None or not SONGS_PATH.exists():
            try:
                log.info("fetching_songs_catalog")
                songs = normalize_songs(fetch(client, "songs.json", key))
                save_json(SONGS_PATH, songs)
                log.info("wrote_songs", count=len(songs))
            except Exception as exc:
                log.warning("songs_catalog_failed", error=str(exc))
                if not SONGS_PATH.exists():
                    save_json(SONGS_PATH, [])
        else:
            log.info("skipping_songs_catalog", reason="file exists on incremental run")

        if year is None or not VENUES_PATH.exists():
            try:
                log.info("fetching_venues")
                venues = normalize_venues(fetch(client, "venues.json", key))
                save_json(VENUES_PATH, venues)
                log.info("wrote_venues", count=len(venues))
            except Exception as exc:
                log.warning("venues_fetch_failed", error=str(exc))
                if not VENUES_PATH.exists():
                    save_json(VENUES_PATH, {})
        else:
            log.info("skipping_venues", reason="file exists on incremental run")


def cli() -> None:
    """Console-script entry point: parse args and call ``main``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=None, help="Single year (else full pull)")
    args = parser.parse_args()
    scrape(args.year)


if __name__ == "__main__":
    cli()
