"""Canonical filesystem paths for data, features, models, and state."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "state" / "features"
MODELS_DIR = ROOT / "models"
STATE_DIR = ROOT / "state"

SETLISTS_PATH = DATA_DIR / "setlists.json"
SONGS_PATH = DATA_DIR / "songs.json"
VENUES_PATH = DATA_DIR / "venues.json"
CANONICAL_NAMES_PATH = ROOT / "data" / "canonical_names.json"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
STATE_SNAPSHOT_PATH = MODELS_DIR / "state.pkl"
