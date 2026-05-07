"""Unit tests for the summarize subpackage."""

from phinish.summarize.types import SummaryResult


def test_summary_result_fields():
    result = SummaryResult(text="Great show!", voice="full_phan", is_fallback=False)
    assert result.text == "Great show!"
    assert result.voice == "full_phan"
    assert result.is_fallback is False


def test_summary_result_fallback():
    result = SummaryResult(text="fallback", voice="light_fan", is_fallback=True)
    assert result.is_fallback is True
