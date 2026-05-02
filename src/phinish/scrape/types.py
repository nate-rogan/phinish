"""Typed shapes for the scrape stage (Phish.net API → ``data/processed/``).

Output Structs (``Show``, ``SongEntry``, etc.) define the persisted schema.
API Structs (``ApiSetlistRow``, ``ApiSongRow``, ``ApiVenueRow``) handle the
raw API response parsing — defaults, coercion, and field-name mapping all live
there so callers never touch ``row.get()``.
"""

import msgspec

# ---------------------------------------------------------------------------
# Output Structs (downstream contract)
# ---------------------------------------------------------------------------


class SongEntry(msgspec.Struct):
    """One song slot inside a set: position, transition mark, jam/reprise flags."""

    song: str
    song_id: str
    position: int
    transition: str
    is_jam: bool
    is_reprise: bool


class Show(msgspec.Struct):
    """One Phish show with date, venue, tour metadata, and all sets played.

    Persisted as one entry in ``data/processed/setlists.json``. Also
    produced in-memory by ``predict.synthetic_show`` for inference.
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
    total_songs: int = 0


class SongCatalogEntry(msgspec.Struct):
    """One row from the Phish.net song catalog (``data/processed/songs.json``)."""

    song_id: str
    name: str
    slug: str
    artist: str
    is_original: bool
    debut: str
    last_played: str
    times_played: int


class VenueRecord(msgspec.Struct):
    """One venue from ``data/processed/venues.json``, keyed by ``venue_id``."""

    venue_id: str
    name: str
    city: str
    state: str
    country: str
    phishnet_id: str = ""


# ---------------------------------------------------------------------------
# API Structs — raw Phish.net API response rows
# ---------------------------------------------------------------------------


class ApiSetlistRow(msgspec.Struct, forbid_unknown_fields=False):
    """One row from ``GET /v5/setlists/showyear/<year>.json``."""

    showid: str = ""
    showdate: str = ""
    venue: str = ""
    city: str = ""
    state: str = ""
    country: str = ""
    tourname: str = ""
    tourid: str = ""
    song: str = ""
    songid: str = ""
    set: str = "1"
    position: int = 0
    trans_mark: str = ""
    transition: str = ""
    isjam: str = "0"
    isreprise: str = "0"

    @property
    def set_key(self) -> str:
        """Normalize encore variants to ``'encore'``."""
        return "encore" if self.set.lower() in ("e", "encore") else self.set

    @property
    def is_jam(self) -> bool:
        return self.isjam == "1"

    @property
    def is_reprise(self) -> bool:
        return self.isreprise == "1"

    @property
    def transition_mark(self) -> str:
        return (self.trans_mark or self.transition or ",").strip() or ","


class ApiSongRow(msgspec.Struct, forbid_unknown_fields=False):
    """One row from ``GET /v5/songs.json``."""

    songid: str = ""
    song: str = ""
    slug: str = ""
    artist: str = ""
    artist_name: str = ""
    debut: str = ""
    last_played: str = ""
    times_played: int = 0
    id: str = ""
    name: str = ""

    @property
    def resolved_id(self) -> str:
        return self.songid or self.id

    @property
    def resolved_name(self) -> str:
        return self.song or self.name

    @property
    def resolved_artist(self) -> str:
        return (self.artist or self.artist_name).strip()


class ApiVenueRow(msgspec.Struct, forbid_unknown_fields=False):
    """One row from ``GET /v5/venues.json``."""

    venueid: str = ""
    venuename: str = ""
    city: str = ""
    state: str = ""
    country: str = ""
    id: str = ""
    name: str = ""

    @property
    def resolved_id(self) -> str:
        return self.venueid or self.id

    @property
    def resolved_name(self) -> str:
        return self.venuename or self.name
