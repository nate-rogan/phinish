"""Anthropic Claude Haiku integration for fan-voiced prediction summaries."""

import os
from pathlib import Path

import httpx
import stamina
import structlog

from phinish.predict.types import Prediction
from phinish.summarize.types import SummaryResult
from phinish.utils import SET_DISPLAY

log = structlog.get_logger()

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 256
TIMEOUT = 10.0

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

FALLBACK_TEXT = (
    "Dude, Anthropic is down, bro... I missed the show. "
    "Check the tables above — the numbers don't lie "
    "even if I can't riff on them right now."
)

VOICE_MAP: dict[str, str] = {
    "Full Phan": "full_phan",
    "Light Fan": "light_fan",
}

DEFAULT_VOICE = "full_phan"


def _load_prompt(voice: str) -> str:
    """Read a prompt template from the prompts directory.

    Parameters
    ----------
    voice
        Template name (without extension). Falls back to ``full_phan``
        if the requested file does not exist.

    Returns
    -------
    str
        Raw template text with ``str.format()`` placeholders.
    """
    path = PROMPTS_DIR / f"{voice}.md"
    if not path.exists():
        log.warning("unknown_voice", voice=voice, fallback=DEFAULT_VOICE)
        path = PROMPTS_DIR / f"{DEFAULT_VOICE}.md"
    return path.read_text(encoding="utf-8")


def _format_setlist(prediction: Prediction) -> str:
    """Render the prediction setlist as plain text for the prompt.

    Parameters
    ----------
    prediction
        Structured prediction payload.

    Returns
    -------
    str
        Human-readable setlist with song names and gap info.
    """
    parts: list[str] = []
    for set_key, label in SET_DISPLAY:
        items = prediction.setlist.get(set_key, [])
        if not items:
            continue
        parts.append(f"{label}:")
        for item in items:
            gap_str = f"{item.gap} show{'s' if item.gap != 1 else ''}"
            parts.append(f"  {item.song} (confidence: {item.confidence:.0%}, gap: {gap_str})")
    return "\n".join(parts)


def _is_retryable(exc: BaseException) -> bool:
    """Retry on 429, 5xx, and network errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.RequestError)


@stamina.retry(on=_is_retryable, attempts=3)
def _call_haiku(system_prompt: str, api_key: str) -> str:
    """Send a single message to Claude Haiku and return the text response.

    Parameters
    ----------
    system_prompt
        Fully rendered system prompt with prediction data injected.
    api_key
        Anthropic API key.

    Returns
    -------
    str
        The model's text response.
    """
    resp = httpx.post(
        API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": system_prompt,
            "messages": [{"role": "user", "content": "Write the preview."}],
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def summarize(prediction: Prediction, voice: str = DEFAULT_VOICE) -> SummaryResult:
    """Generate a fan-voiced summary of a setlist prediction via Claude Haiku.

    Parameters
    ----------
    prediction
        The structured prediction (sets, songs, confidence scores).
    voice
        Prompt template name or dropdown label. Maps to a file in ``prompts/``.

    Returns
    -------
    SummaryResult
        Contains the summary text and whether it came from the LLM or the
        static fallback.
    """
    resolved_voice = VOICE_MAP.get(voice, voice)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning("missing_anthropic_key", action="using fallback")
        return SummaryResult(text=FALLBACK_TEXT, voice=resolved_voice, is_fallback=True)

    template = _load_prompt(resolved_voice)
    city_part = f", {prediction.city}" if prediction.city else ""
    system_prompt = template.format(
        venue=prediction.venue,
        city=city_part,
        date=prediction.date,
        setlist=_format_setlist(prediction),
        confidence=f"{prediction.avg_confidence:.0%}",
    )

    try:
        text = _call_haiku(system_prompt, api_key)
    except Exception:
        log.exception("haiku_call_failed", action="using fallback")
        return SummaryResult(text=FALLBACK_TEXT, voice=resolved_voice, is_fallback=True)

    return SummaryResult(text=text, voice=resolved_voice, is_fallback=False)
