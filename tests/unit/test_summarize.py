"""Unit tests for the summarize subpackage."""

from pathlib import Path

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
