"""Train order-2 Markov chain over the full dataset.

Writes models/markov_order2.json. Same shape as state/features/transition_matrix.json
but versioned in models/ as the inference artifact.
"""

import structlog

from phinish.artifacts import load_shows
from phinish.features.build import build_transition_matrix
from phinish.features.types import TransitionMatrix
from phinish.utils import MODELS_DIR, SETLISTS_PATH, save_json

log = structlog.get_logger()


def train_markov(shows: list) -> TransitionMatrix:
    """Train the order-2 Markov transition matrix from show history.

    Parameters
    ----------
    shows
        List of ``Show`` structs.

    Returns
    -------
    TransitionMatrix
        Trained transition matrix with ``trained_on_shows`` set.
    """
    matrix = build_transition_matrix(shows)
    matrix.trained_on_shows = len(shows)
    return matrix


def main() -> None:
    """Console entry point: load shows, train Markov model, write to disk."""
    if not SETLISTS_PATH.exists():
        raise SystemExit(f"Missing {SETLISTS_PATH}; run phinish-scrape first.")
    matrix = train_markov(load_shows())
    save_json(MODELS_DIR / "markov_order2.json", matrix)
    log.info(
        "wrote_markov", shows=matrix.trained_on_shows, path=str(MODELS_DIR / "markov_order2.json")
    )


if __name__ == "__main__":
    main()
