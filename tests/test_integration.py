"""Integration tests: data → features pipeline and last-show exclusion."""

import json

from scripts import build_features
from scripts.train_xgboost import StreamingState


def test_data_to_features_pipeline(tmp_path, monkeypatch, tiny_shows, songs_catalog):
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    features_dir = tmp_path / "state" / "features"

    setlists_path = data_dir / "setlists.json"
    songs_path = data_dir / "songs.json"
    setlists_path.write_text(json.dumps(tiny_shows), encoding="utf-8")
    songs_path.write_text(json.dumps(songs_catalog), encoding="utf-8")

    monkeypatch.setattr(build_features, "SETLISTS_PATH", setlists_path)
    monkeypatch.setattr(build_features, "SONGS_PATH", songs_path)
    monkeypatch.setattr(build_features, "FEATURES_DIR", features_dir)

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
