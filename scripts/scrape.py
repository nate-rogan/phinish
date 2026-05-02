"""Phish.net API v5 scraper.

Pulls setlists, songs, venues and writes:
  data/processed/setlists.json
  data/processed/songs.json
  data/processed/venues.json

Usage:
  pixi run python scripts/scrape.py              # full pull (1983-present)
  pixi run python scripts/scrape.py --year 2025  # one year, merged into existing

Requires PHISHNET_API_KEY environment variable.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

import httpx

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils import (
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

API_BASE = "https://api.phish.net/v5"
FIRST_YEAR = 1983
TIMEOUT = 30.0
RETRIES = 3
RETRY_BACKOFF_BASE = 2.0


def get_api_key() -> str:
    """Read PHISHNET_API_KEY from the environment or exit with a clear message."""
    key = os.environ.get("PHISHNET_API_KEY")
    if not key:
        raise SystemExit("PHISHNET_API_KEY environment variable is required.")
    return key


def fetch(client: httpx.Client, path: str, key: str) -> list[dict]:
    """GET an API path and return its `data` payload, retrying transient errors."""
    url = f"{API_BASE}/{path.lstrip('/')}"
    last_err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = client.get(url, params={"apikey": key}, timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if payload.get("error"):
                raise RuntimeError(f"{path}: {payload.get('error_message')}")
            return payload.get("data", []) or []
        except (httpx.HTTPError, RuntimeError) as e:
            last_err = e
            if attempt < RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
    raise RuntimeError(f"Failed after {RETRIES} attempts: {path}: {last_err}")


def group_into_shows(rows: list[dict], canonical: dict[str, str]) -> list[dict]:
    """Collapse Phish.net per-song rows into per-show records sorted by date."""
    shows: dict[str, dict] = {}
    for row in rows:
        sid = str(row.get("showid", ""))
        if not sid:
            continue
        if sid not in shows:
            d = str(row.get("showdate", ""))
            shows[sid] = {
                "show_id": sid,
                "date": d,
                "year": int(d[:4]) if len(d) >= 4 else 0,
                "month": int(d[5:7]) if len(d) >= 7 else 0,
                "day": int(d[8:10]) if len(d) >= 10 else 0,
                "day_of_week": day_of_week(d) if len(d) == 10 else "",
                "venue_id": venue_id(row.get("venue", "")),
                "venue_name": row.get("venue", "") or "",
                "city": row.get("city", "") or "",
                "state": row.get("state", "") or "",
                "country": row.get("country", "") or "",
                "tour": row.get("tourname", "") or "",
                "tour_id": str(row.get("tourid", "") or ""),
                "sets": {},
            }
        show = shows[sid]
        set_raw = str(row.get("set", "1"))
        set_key = "encore" if set_raw.lower() in ("e", "encore") else set_raw
        show["sets"].setdefault(set_key, []).append({
            "song": canonicalize_song(row.get("song", ""), canonical),
            "song_id": str(row.get("songid", "") or ""),
            "position": int(row.get("position", 0) or 0),
            "transition": (row.get("trans_mark") or row.get("transition") or ",").strip() or ",",
            "is_jam": str(row.get("isjam", "0")) == "1",
            "is_reprise": str(row.get("isreprise", "0")) == "1",
        })

    out: list[dict] = []
    for show in shows.values():
        cleaned = {k: sorted(v, key=lambda s: s["position"]) for k, v in show["sets"].items() if v}
        show["sets"] = cleaned
        show["total_songs"] = sum(len(v) for v in cleaned.values())
        show.update(special_show_flags(show["date"], show["tour"]))
        out.append(show)
    out.sort(key=lambda s: s["date"])
    return out


def merge_setlists(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge new shows into existing, replacing duplicates by show_id."""
    by_id = {s["show_id"]: s for s in existing}
    for s in new:
        by_id[s["show_id"]] = s
    return sorted(by_id.values(), key=lambda s: s["date"])


def normalize_songs(rows: list[dict]) -> list[dict]:
    """Reshape Phish.net song catalog rows into the project's schema."""
    out = []
    for r in rows:
        artist = (r.get("artist") or r.get("artist_name") or "").strip()
        out.append({
            "song_id": str(r.get("songid") or r.get("id") or ""),
            "name": r.get("song") or r.get("name") or "",
            "slug": r.get("slug", ""),
            "artist": artist,
            "is_original": artist.lower() == "phish",
            "debut": r.get("debut", "") or "",
            "last_played": r.get("last_played", "") or "",
            "times_played": int(r.get("times_played", 0) or 0),
        })
    return out


def normalize_venues(rows: list[dict]) -> dict[str, dict]:
    """Reshape Phish.net venue rows into the project's `{venue_id: ...}` schema."""
    venues: dict[str, dict] = {}
    for r in rows:
        name = r.get("venuename") or r.get("name") or ""
        if not name:
            continue
        vid = venue_id(name)
        venues[vid] = {
            "venue_id": vid,
            "phishnet_id": str(r.get("venueid") or r.get("id") or ""),
            "name": name,
            "city": r.get("city", "") or "",
            "state": r.get("state", "") or "",
            "country": r.get("country", "") or "",
        }
    return venues


def main(year: int | None) -> None:
    """Pull setlists/songs/venues from Phish.net and persist to data/processed/."""
    key = get_api_key()
    canonical = load_json(CANONICAL_NAMES_PATH) if CANONICAL_NAMES_PATH.exists() else {}
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        if year is None:
            years = range(FIRST_YEAR, date.today().year + 1)
            all_rows: list[dict] = []
            for y in years:
                print(f"  fetching setlists for {y}...", flush=True)
                all_rows.extend(fetch(client, f"setlists/showyear/{y}.json", key))
            shows = group_into_shows(all_rows, canonical)
            save_json(SETLISTS_PATH, shows)
            print(f"wrote {len(shows)} shows -> {SETLISTS_PATH}")
        else:
            print(f"  fetching setlists for {year}...", flush=True)
            rows = fetch(client, f"setlists/showyear/{year}.json", key)
            new_shows = group_into_shows(rows, canonical)
            existing = load_json(SETLISTS_PATH) if SETLISTS_PATH.exists() else []
            merged = merge_setlists(existing, new_shows)
            save_json(SETLISTS_PATH, merged)
            print(f"merged {len(new_shows)} shows for {year}; total {len(merged)}")

        print("  fetching songs catalog...", flush=True)
        songs = normalize_songs(fetch(client, "songs.json", key))
        save_json(SONGS_PATH, songs)
        print(f"wrote {len(songs)} songs")

        print("  fetching venues...", flush=True)
        venues = normalize_venues(fetch(client, "venues.json", key))
        save_json(VENUES_PATH, venues)
        print(f"wrote {len(venues)} venues")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=None, help="Single year (else full pull)")
    args = parser.parse_args()
    main(args.year)
