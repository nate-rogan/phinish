"""Train order-2 Markov chain over the full dataset.

Writes models/markov_order2.json. Same shape as state/features/transition_matrix.json
but versioned in models/ as the inference artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_features import build_transition_matrix
from scripts.utils import MODELS_DIR, SETLISTS_PATH, load_json, save_json


def main() -> None:
    """Train and persist the order-2 Markov transition matrix to models/."""
    if not SETLISTS_PATH.exists():
        raise SystemExit(f"Missing {SETLISTS_PATH}; run scrape.py first.")
    shows = load_json(SETLISTS_PATH)
    matrix = build_transition_matrix(shows)
    matrix["trained_on_shows"] = len(shows)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(MODELS_DIR / "markov_order2.json", matrix)
    print(f"wrote markov ({len(shows)} shows) -> {MODELS_DIR / 'markov_order2.json'}")


if __name__ == "__main__":
    main()
