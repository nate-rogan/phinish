"""Unit tests for build_features."""

import msgspec

from phinish.features.build import (
    build_song_gaps,
    build_song_stats,
    build_transition_matrix,
)
from phinish.scrape.types import SongCatalogEntry


def test_song_gap_against_hand_verified(tiny_shows):
    gaps = build_song_gaps(tiny_shows)
    # Total = 5 shows; gap = total - 1 - last_idx
    # A last in show 4 (idx 3) -> gap 1
    # B last in show 5 (idx 4) -> gap 0
    # D last in show 3 (idx 2) -> gap 2
    # E last in show 2 (idx 1) -> gap 3
    assert gaps["A"].gap == 1
    assert gaps["B"].gap == 0
    assert gaps["D"].gap == 2
    assert gaps["E"].gap == 3
    assert gaps["A"].last_played == "2024-01-04"


def test_transition_probabilities_smoothed(tiny_shows):
    matrix = build_transition_matrix(tiny_shows)
    # Within sets: A->B occurs in shows 1, 2, and 4 (set 1: A,B / A,B / A,F)
    # Show 1 set 1: A->B
    # Show 2 set 1: A->B
    # Show 4 set 1: A->F
    a_dist = matrix.order_1["A"]
    # Each emitted prob in (0, 1); B should outweigh F from A.
    assert all(0 < p < 1 for p in a_dist.values())
    assert a_dist["B"] > a_dist["F"]
    # Sum of *stored* probs is just under 1 (Laplace mass for unseen songs not stored).
    assert sum(a_dist.values()) <= 1.0
    assert sum(a_dist.values()) > 0.95


def test_song_stats_schema(tiny_shows, songs_catalog):
    stats = build_song_stats(tiny_shows, songs_catalog)
    required = {
        "total_plays", "lifetime_frequency", "recent_frequency_50",
        "recent_frequency_20", "avg_set_position", "typical_set",
        "set_distribution", "opener_frequency", "closer_frequency",
        "is_cover", "debut_year",
    }
    for song, s in stats.items():
        actual = {f.name for f in msgspec.structs.fields(s)}
        missing = required - actual
        assert not missing, f"{song} missing keys {missing}"
        assert 0.0 <= s.lifetime_frequency <= 1.0
        assert 0.0 <= s.opener_frequency <= 1.0
        assert s.typical_set in (1, 2, 3)


def test_song_with_zero_plays_excluded(tiny_shows, songs_catalog):
    extended_catalog = [
        *songs_catalog,
        SongCatalogEntry(
            song_id="g", name="G", slug="g", artist="Phish",
            is_original=True, debut="", last_played="", times_played=0,
        ),
    ]
    stats = build_song_stats(tiny_shows, extended_catalog)
    assert "G" not in stats
