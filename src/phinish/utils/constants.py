"""Domain constants and regex patterns shared across stages."""

import re

VALID_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VALID_YEAR = re.compile(r"^\d{4}$")

SET_KEYS: tuple[str, ...] = ("1", "2", "3", "encore")
SET_DISPLAY: tuple[tuple[str, str], ...] = (("1", "Set 1"), ("2", "Set 2"), ("encore", "Encore"))
SET_TO_INT: dict[str, int] = {"1": 1, "2": 2, "3": 3, "encore": 3}
TOP_K: int = 25  # prediction window; matches typical show song count
FUZZY_VENUE_THRESHOLD: float = 0.7  # SequenceMatcher ratio; below this, matches are unreliable
