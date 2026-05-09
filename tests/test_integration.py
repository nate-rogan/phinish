"""Integration tests: data → features pipeline and last-show exclusion."""

import json

import msgspec
import pytest

from phinish import artifacts
from phinish.features import build as build_features
from phinish.features.build import build_song_gaps
from phinish.train import MIN_PLAYS_FOR_CANDIDATE, StreamingState
from phinish.utils import SETLISTS_PATH, show_song_set


def test_data_to_features_pipeline(tmp_path, monkeypatch, tiny_shows, songs_catalog):
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    features_dir = tmp_path / "state" / "features"

    setlists_path = data_dir / "setlists.json"
    songs_path = data_dir / "songs.json"
    setlists_path.write_bytes(msgspec.json.encode(tiny_shows))
    songs_path.write_bytes(msgspec.json.encode(songs_catalog))

    monkeypatch.setattr(build_features, "SETLISTS_PATH", setlists_path)
    monkeypatch.setattr(build_features, "FEATURES_DIR", features_dir)
    monkeypatch.setattr(artifacts, "SETLISTS_PATH", setlists_path)
    monkeypatch.setattr(artifacts, "SONGS_PATH", songs_path)

    build_features.main()

    expected = ("song_gaps.json", "song_stats.json", "transition_matrix.json", "venue_history.json")
    for name in expected:
        path = features_dir / name
        assert path.exists(), f"missing {name}"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data, f"{name} is empty"

    gaps = json.loads((features_dir / "song_gaps.json").read_text(encoding="utf-8"))
    # Last show (2024-01-05) had B and C; their gap should be 0
    assert gaps["B"]["gap"] == 0
    assert gaps["C"]["gap"] == 0


def test_streaming_state_tracks_last_show(tiny_shows):
    """state.last_show_set holds the most recent show's songs (used to exclude them)."""
    state = StreamingState()
    for show in tiny_shows:
        state.update(show)
    assert state.last_show_set == {"B", "C"}
    # Songs from the last show should never be ranked in a fresh prediction
    candidates = ["A", "B", "C", "D", "E", "F"]
    eligible = [s for s in candidates if s not in state.last_show_set]
    assert "B" not in eligible and "C" not in eligible
    assert {"A", "D", "E", "F"} <= set(eligible)


def test_streaming_state_last_3_window(tiny_shows):
    state = StreamingState()
    for show in tiny_shows:
        state.update(show)
    assert len(state.last_3_shows) == 3
    last_3_union = set().union(*state.last_3_shows)
    # Shows 3, 4, 5 had: {C,D}, {A,F}, {B,C}
    assert last_3_union == {"A", "B", "C", "D", "F"}


@pytest.mark.skipif(not SETLISTS_PATH.exists(), reason="no live data")
class TestLiveDataConsistency:
    """Smoke tests against the committed setlists.json to catch data/model drift."""

    def test_gaps_match_actual_show_distance(self):
        """Every gap in the feature store must equal total_shows - 1 - last_index."""
        shows = artifacts.load_shows()
        gaps = build_song_gaps(shows)
        stored_gaps = artifacts.load_song_gaps()
        for song, stored in stored_gaps.items():
            computed = gaps.get(song)
            assert computed is not None, f"{song!r} in feature store but not in setlists"
            assert stored.gap == computed.gap, (
                f"{song!r}: stored gap={stored.gap}, computed gap={computed.gap}"
            )

    def test_candidates_have_gap_entries(self):
        """Every candidate song in the streaming state must have a gap entry."""
        shows = artifacts.load_shows()
        gaps = build_song_gaps(shows)
        state = StreamingState()
        for show in shows:
            state.update(show)
        candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
        missing = [s for s in candidates if s not in gaps]
        assert not missing, f"Candidates without gaps: {missing}"

    def test_no_candidate_has_zero_plays_in_setlists(self):
        """Candidates must actually appear in the setlist data."""
        shows = artifacts.load_shows()
        all_played: set[str] = set()
        for show in shows:
            all_played |= show_song_set(show)
        state = StreamingState()
        for show in shows:
            state.update(show)
        candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
        phantom = [s for s in candidates if s not in all_played]
        assert not phantom, f"Candidates never played in setlists: {phantom}"

    def test_max_year_matches_dataset(self):
        """The dataset's max year should match what split_years would use."""
        shows = artifacts.load_shows()
        max_year = max(s.year for s in shows)
        assert max_year == shows[-1].year, "Shows not sorted chronologically"
