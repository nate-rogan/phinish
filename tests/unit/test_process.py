"""Tests for the process/year model card renderer."""

import pytest

from phinish.process.year import _metric_rows, _render_card, _training_card_section


@pytest.fixture()
def training():
    return {
        "n_train_rows": 1000,
        "n_val_shows": 10,
        "val_year": 1999,
        "val_precision_at_25": 0.35,
        "ensemble_weights": {"xgboost": 0.33, "markov": 0.0, "gap": 0.0, "venue": 0.67},
    }


@pytest.fixture()
def summary():
    return {
        "ensemble": {
            "precision_at_25": 0.32, "recall": 0.56, "f1": 0.39,
            "opener_accuracy": 0.04, "n_shows": 108,
        },
        "frequency": {
            "precision_at_25": 0.06, "recall": 0.09, "f1": 0.07,
            "opener_accuracy": 0.0, "n_shows": 108,
        },
    }


def test_render_card_uses_dynamic_val_year(training, summary):
    card = _render_card("2026-01-01", 1464, "year 2000", _metric_rows(summary), training, 2000)
    assert "validation set (1999)" in card
    assert "2024" not in card


def test_render_card_uses_dynamic_test_year(training, summary):
    card = _render_card("2026-01-01", 1464, "year 2000", _metric_rows(summary), training, 2000)
    assert "test >= 2000" in card


def test_training_card_section_shows_val_year(training):
    section = _training_card_section(training)
    assert "year 1999" in section
    assert "Val Precision@25" in section
