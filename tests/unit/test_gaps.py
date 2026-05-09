"""Tests for gap computation and prediction consistency."""

import pytest

from phinish.features.build import build_song_gaps
from phinish.train.state import MIN_PLAYS_FOR_CANDIDATE, StreamingState
from phinish.utils import show_song_set

from conftest import _show


def _many_shows() -> list:
    """Build a dataset where songs A-E each have >= MIN_PLAYS_FOR_CANDIDATE plays."""
    songs = ["A", "B", "C", "D", "E"]
    shows = []
    for i in range(10):
        d = f"2024-01-{i + 1:02d}"
        shows.append(_show(d, "Venue", {"1": songs[:3], "2": songs[3:]}))
    # Add a final show with only A and B to create known gaps
    shows.append(_show("2024-01-11", "Venue", {"1": ["A", "B"]}))
    return shows


class TestGapComputation:
    """Verify gap = total_shows - 1 - last_show_index."""

    def test_gap_zero_for_most_recent(self):
        shows = _many_shows()
        gaps = build_song_gaps(shows)
        assert gaps["A"].gap == 0
        assert gaps["B"].gap == 0

    def test_gap_one_for_penultimate(self):
        shows = _many_shows()
        gaps = build_song_gaps(shows)
        # C, D, E last played in show index 9 (2024-01-10), total=11
        assert gaps["C"].gap == 1
        assert gaps["D"].gap == 1
        assert gaps["E"].gap == 1

    def test_gap_matches_show_count(self, tiny_shows):
        gaps = build_song_gaps(tiny_shows)
        total = len(tiny_shows)
        for song, gap_obj in gaps.items():
            # Verify gap is non-negative and < total shows
            assert 0 <= gap_obj.gap < total, f"{song}: gap {gap_obj.gap} out of range"

    def test_last_played_date_correct(self):
        shows = _many_shows()
        gaps = build_song_gaps(shows)
        assert gaps["A"].last_played == "2024-01-11"
        assert gaps["C"].last_played == "2024-01-10"


class TestCandidateFiltering:
    """Songs with fewer than MIN_PLAYS_FOR_CANDIDATE plays must not be candidates."""

    def test_rare_songs_excluded(self):
        shows = [
            _show("2024-01-01", "V", {"1": ["Common", "Rare"]}),
            _show("2024-01-02", "V", {"1": ["Common"]}),
            _show("2024-01-03", "V", {"1": ["Common"]}),
            _show("2024-01-04", "V", {"1": ["Common"]}),
            _show("2024-01-05", "V", {"1": ["Common"]}),
        ]
        state = StreamingState()
        for s in shows:
            state.update(s)
        candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
        assert "Common" in candidates
        assert "Rare" not in candidates

    def test_unplayed_song_not_candidate(self):
        state = StreamingState()
        candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
        assert candidates == []


class TestGapPredictionConsistency:
    """Gaps in predictions must match the feature store values."""

    def test_gaps_only_for_known_songs(self):
        shows = _many_shows()
        gaps = build_song_gaps(shows)
        state = StreamingState()
        for s in shows:
            state.update(s)
        candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
        for song in candidates:
            assert song in gaps, f"Candidate {song!r} has no gap entry"

    def test_all_candidates_were_played(self):
        shows = _many_shows()
        state = StreamingState()
        all_played = set()
        for s in shows:
            all_played |= show_song_set(s)
            state.update(s)
        candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
        for song in candidates:
            assert song in all_played, f"Candidate {song!r} was never played"

    @pytest.mark.parametrize("gap_threshold", [0, 1, 5, 50])
    def test_gap_never_negative(self, gap_threshold):
        shows = _many_shows()
        gaps = build_song_gaps(shows)
        for song, gap_obj in gaps.items():
            assert gap_obj.gap >= 0, f"{song}: negative gap {gap_obj.gap}"
