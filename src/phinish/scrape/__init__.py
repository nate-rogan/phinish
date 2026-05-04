"""Scrape stage: pull setlists / songs / venues from the Phish.net API."""

from phinish.scrape.api import cli, scrape
from phinish.scrape.types import Show, SongCatalogEntry, SongEntry, VenueRecord

__all__ = ("Show", "SongCatalogEntry", "SongEntry", "VenueRecord", "cli", "scrape")
