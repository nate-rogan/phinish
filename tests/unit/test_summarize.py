"""Unit tests for the summarize subpackage."""

from pathlib import Path

import httpx
import pytest

from phinish.predict.types import Prediction, PredictionItem, PredictionWeights
from phinish.summarize.api import FALLBACK_TEXT, VOICE_MAP, _format_setlist, _load_prompt, summarize
from phinish.summarize.types import SummaryResult

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "src" / "phinish" / "summarize" / "prompts"


def test_summary_result_fields():
    result = SummaryResult(text="Great show!", voice="full_phan", is_fallback=False)
    assert result.text == "Great show!"
    assert result.voice == "full_phan"
    assert result.is_fallback is False


def test_summary_result_fallback():
    result = SummaryResult(text="fallback", voice="light_fan", is_fallback=True)
    assert result.is_fallback is True


def test_prompt_templates_exist():
    for name in ("full_phan", "light_fan"):
        path = PROMPTS_DIR / f"{name}.md"
        assert path.exists(), f"Missing prompt template: {path}"


def test_prompt_templates_have_required_placeholders():
    required = {"{venue}", "{date}", "{setlist}", "{confidence}"}
    for name in ("full_phan", "light_fan"):
        content = (PROMPTS_DIR / f"{name}.md").read_text()
        for placeholder in required:
            assert placeholder in content, f"{name}.md missing {placeholder}"


def test_load_prompt_full_phan():
    text = _load_prompt("full_phan")
    assert "die-hard Phish fan" in text
    assert "{venue}" in text


def test_load_prompt_unknown_voice_falls_back():
    text = _load_prompt("nonexistent_voice")
    assert "{venue}" in text  # falls back to full_phan


def test_voice_map_covers_dropdown_labels():
    assert VOICE_MAP["Full Phan"] == "full_phan"
    assert VOICE_MAP["Light Fan"] == "light_fan"


@pytest.fixture()
def sample_prediction():
    return Prediction(
        date="2026-12-31", venue="MSG", venue_id="v_msg", city="New York, NY",
        avg_confidence=0.64, model_version=1,
        weights=PredictionWeights(xgboost=0.5, markov=0.1, gap=0.3, venue=0.1),
        setlist={
            "1": [PredictionItem(song="Tweezer", confidence=0.72, gap=15),
                  PredictionItem(song="Divided Sky", confidence=0.60, gap=8)],
            "2": [PredictionItem(song="Disease", confidence=0.55, gap=5)],
            "encore": [PredictionItem(song="Character Zero", confidence=0.50, gap=4)],
        },
    )


def test_format_setlist(sample_prediction):
    text = _format_setlist(sample_prediction)
    assert "Set 1:" in text
    assert "Tweezer" in text
    assert "15 shows" in text
    assert "Encore:" in text


def test_summarize_returns_fallback_when_no_api_key(monkeypatch, sample_prediction):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = summarize(sample_prediction, voice="full_phan")
    assert result.is_fallback is True
    assert result.text == FALLBACK_TEXT
    assert result.voice == "full_phan"


def test_summarize_returns_fallback_on_http_error(monkeypatch, sample_prediction):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def mock_post(*args, **kwargs):
        raise httpx.HTTPStatusError(
            "Server Error", request=httpx.Request("POST", "https://x"), response=httpx.Response(500)
        )

    monkeypatch.setattr(httpx, "post", mock_post)
    result = summarize(sample_prediction, voice="full_phan")
    assert result.is_fallback is True


def test_summarize_returns_llm_text_on_success(monkeypatch, sample_prediction):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def mock_post(*args, **kwargs):
        resp = httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "Dude, Tweezer is raging tonight!"}]},
            request=httpx.Request("POST", "https://x"),
        )
        return resp

    monkeypatch.setattr(httpx, "post", mock_post)
    result = summarize(sample_prediction, voice="full_phan")
    assert result.is_fallback is False
    assert result.text == "Dude, Tweezer is raging tonight!"
    assert result.voice == "full_phan"
