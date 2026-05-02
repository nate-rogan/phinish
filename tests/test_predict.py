"""Unit tests for predict.py helpers and output structure."""

import pytest

from phinish.predict import _split_by_typical_set, predict, synthetic_show
from phinish.process_issue import format_comment
from phinish.utils import min_max_normalize


@pytest.mark.parametrize("vals,expected", [
    pytest.param([1.0, 2.0, 3.0], [0.0, 0.5, 1.0], id="three_values"),
    pytest.param([5.0], [0.0], id="single_value_collapses_to_zero"),
    pytest.param([], [], id="empty"),
    pytest.param([2.0, 2.0, 2.0], [0.0, 0.0, 0.0], id="constant_input"),
])
def test_norm_to_unit_range(vals, expected):
    assert min_max_normalize(vals) == expected


def test_split_by_typical_set():
    stats = {
        "A": {"typical_set": 1},
        "B": {"typical_set": 2},
        "C": {"typical_set": 3},
        "D": {"typical_set": 1},
    }
    buckets = _split_by_typical_set(stats, ["A", "B", "C", "D"])
    assert "A" in buckets["1"] and "D" in buckets["1"]
    assert buckets["2"] == ["B"]
    assert buckets["encore"] == ["C"]


@pytest.mark.parametrize("bad_date", ["2026/12/31", "12-31-2026", "not-a-date", "", "2026-13-01x"])
def test_predict_rejects_invalid_date(bad_date):
    with pytest.raises(ValueError, match="Invalid date format"):
        predict(bad_date, "MSG")


def test_synthetic_show_resolves_venue():
    venues = {"v_msg": {"venue_id": "v_msg", "name": "Madison Square Garden",
                        "city": "New York", "state": "NY", "country": "USA"}}
    show = synthetic_show("2026-12-31", "Madison Square Garden", venues)
    assert show["date"] == "2026-12-31"
    assert show["venue_id"] == "v_msg"
    assert show["is_nye"] is True
    assert show["year"] == 2026
    assert show["month"] == 12
    assert show["day"] == 31


def test_format_comment_well_formed():
    prediction = {
        "date": "2026-12-31", "venue": "MSG", "city": "NY",
        "avg_confidence": 0.42,
        "weights": {"xgboost": 0.5, "markov": 0.1, "gap": 0.3, "venue": 0.1},
        "setlist": {
            "1": [{"song": "Buried Alive", "confidence": 0.72, "gap": 5}],
            "2": [{"song": "Tweezer", "confidence": 0.68, "gap": 3}],
            "encore": [{"song": "Character Zero", "confidence": 0.55, "gap": 4}],
        },
    }
    md = format_comment(prediction)
    assert "## 🎸 Phinish Prediction" in md
    assert "MSG" in md and "2026-12-31" in md
    for song in ("Buried Alive", "Tweezer", "Character Zero"):
        assert song in md
    assert "72%" in md


def test_prediction_setlist_constraints():
    """No duplicate songs and all confidences in [0,1]."""
    setlist = {
        "1": [{"song": "A", "confidence": 0.7, "gap": 1},
              {"song": "B", "confidence": 0.6, "gap": 2}],
        "2": [{"song": "C", "confidence": 0.5, "gap": 3}],
        "encore": [{"song": "D", "confidence": 0.4, "gap": 4}],
    }
    seen = set()
    for songs in setlist.values():
        for s in songs:
            assert 0.0 <= s["confidence"] <= 1.0
            assert s["song"] not in seen
            seen.add(s["song"])
