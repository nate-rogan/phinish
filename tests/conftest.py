"""Shared pytest fixtures: small synthetic setlist data."""
from __future__ import annotations

import pytest

from phinish.utils import Show, SongCatalogEntry, SongEntry


def _song(song: str, position: int) -> SongEntry:
    return {
        "song": song,
        "song_id": song.lower(),
        "position": position,
        "transition": ",",
        "is_jam": False,
        "is_reprise": False,
    }


def _show(show_date: str, venue: str, sets: dict[str, list[str]]) -> Show:
    return {
        "show_id": show_date,
        "date": show_date,
        "year": int(show_date[:4]),
        "month": int(show_date[5:7]),
        "day": int(show_date[8:10]),
        "day_of_week": "saturday",
        "venue_id": "v_" + venue.lower().replace(" ", "_"),
        "venue_name": venue,
        "city": "",
        "state": "",
        "country": "USA",
        "tour": "Test Tour",
        "tour_id": "t1",
        "sets": {k: [_song(s, i) for i, s in enumerate(songs)] for k, songs in sets.items()},
        "is_nye": False,
        "is_halloween": False,
        "is_festival": False,
        "total_songs": sum(len(v) for v in sets.values()),
    }


@pytest.fixture
def tiny_shows() -> list[Show]:
    """Five shows with known song positions for hand-verifying features."""
    return [
        _show("2024-01-01", "Fake Venue", {"1": ["A", "B"], "2": ["C", "D"]}),
        _show("2024-01-02", "Fake Venue", {"1": ["A", "B"], "2": ["E"]}),
        _show("2024-01-03", "Other Venue", {"1": ["C", "D"]}),
        _show("2024-01-04", "Fake Venue", {"1": ["A", "F"]}),
        _show("2024-01-05", "Fake Venue", {"1": ["B", "C"]}),
    ]


@pytest.fixture
def songs_catalog() -> list[SongCatalogEntry]:
    """Minimal song catalog matching the songs used in `tiny_shows`."""
    return [
        {
            "song_id": s.lower(), "name": s, "slug": s.lower(), "artist": "Phish",
            "is_original": True, "debut": "2024-01-01", "last_played": "",
            "times_played": 0,
        }
        for s in ("A", "B", "C", "D", "E", "F")
    ]
