"""Tests for the prompts loader (DESIGN.md §5.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.prompts import Prompt, load_prompts

REPO_ROOT = Path(__file__).resolve().parents[3]
DESIGN_PROMPTS_MD = REPO_ROOT / "docs" / "prompts.md"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# The six (brand, model) target pairs from DESIGN.md §1.2.
TARGET_PAIRS = {
    ("Gibson", "Les Paul"),
    ("Gibson", "SG"),
    ("Gibson", "Explorer"),
    ("Gibson", "Flying V"),
    ("Fender", "Stratocaster"),
    ("Fender", "Telecaster"),
}


def test_load_design_prompts_returns_nine_entries():
    prompts = load_prompts(DESIGN_PROMPTS_MD)
    # Regression guard against accidental prompt deletion.
    # When adding a new target, bump this assertion deliberately.
    assert len(prompts) == 9
    assert all(isinstance(p, Prompt) for p in prompts)


def test_design_prompts_cover_six_target_models():
    prompts = load_prompts(DESIGN_PROMPTS_MD)
    non_unknown = {
        (p.brand, p.model) for p in prompts if p.brand != "Unknown" and p.model != "Unknown"
    }
    assert non_unknown == TARGET_PAIRS


def test_design_prompts_have_three_unknown_rejection_entries():
    prompts = load_prompts(DESIGN_PROMPTS_MD)
    unknowns = [p for p in prompts if p.brand == "Unknown" and p.model == "Unknown"]
    assert len(unknowns) == 3


def test_load_minimal_yaml_fixture():
    prompts = load_prompts(FIXTURES_DIR / "prompts_minimal.yaml")
    assert len(prompts) == 2
    assert prompts[0].text == "a photograph of a thing"
    assert prompts[0].brand == "Foo"
    assert prompts[0].model == "Bar"
    assert prompts[1].text == "a photograph of another thing"
    assert prompts[1].brand == "Baz"
    assert prompts[1].model == "Qux"


def test_load_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_prompts(missing)


def test_load_md_without_yaml_block_raises_valueerror(tmp_path: Path):
    md = tmp_path / "no_yaml.md"
    md.write_text("# Heading\n\nNo code fence here at all.\n")
    with pytest.raises(ValueError, match="yaml"):
        load_prompts(md)


def test_load_malformed_yaml_raises_valueerror(tmp_path: Path):
    bad = tmp_path / "broken.yaml"
    bad.write_text('prompts:\n  - text: "unclosed\n')
    with pytest.raises(ValueError):
        load_prompts(bad)


def test_load_yaml_missing_prompts_key_raises_valueerror(tmp_path: Path):
    bad = tmp_path / "wrong_key.yaml"
    bad.write_text("not_prompts: []\n")
    with pytest.raises(ValueError, match="prompts"):
        load_prompts(bad)


def test_load_yaml_prompt_missing_field_raises_valueerror(tmp_path: Path):
    bad = tmp_path / "missing_field.yaml"
    bad.write_text(
        'prompts:\n  - text: "a photograph of a thing"\n    model: Bar\n',
    )
    with pytest.raises(ValueError):
        load_prompts(bad)


def test_load_yaml_empty_prompts_list_raises_valueerror(tmp_path: Path):
    """An empty prompts list is rejected (downstream classifier cannot run)."""
    p = tmp_path / "empty.yaml"
    p.write_text("prompts: []\n")
    with pytest.raises(ValueError, match="empty"):
        load_prompts(p)


def test_load_yaml_whitespace_only_field_raises_valueerror(tmp_path: Path):
    """A prompt with a whitespace-only required field is rejected."""
    p = tmp_path / "whitespace.yaml"
    p.write_text('prompts:\n  - text: "   "\n    brand: Gibson\n    model: Les Paul\n')
    with pytest.raises(ValueError):
        load_prompts(p)


def test_load_yaml_unknown_field_raises_valueerror(tmp_path: Path):
    """Unknown fields on a prompt entry are rejected (catches typos)."""
    p = tmp_path / "extra.yaml"
    p.write_text(
        "prompts:\n"
        "  - text: ok\n"
        "    brand: Gibson\n"
        "    model: Les Paul\n"
        "    tags: [vintage]\n"
    )
    with pytest.raises(ValueError):
        load_prompts(p)


def test_load_md_with_multiple_yaml_blocks_picks_first(tmp_path: Path):
    """When a Markdown file contains multiple ```yaml blocks, only the first is parsed."""
    md = tmp_path / "many.md"
    md.write_text(
        "# Header\n"
        "```yaml\n"
        "prompts:\n"
        "  - text: t1\n"
        "    brand: B1\n"
        "    model: M1\n"
        "```\n"
        "Some prose.\n"
        "```yaml\n"
        "prompts:\n"
        "  - text: t2\n"
        "    brand: B2\n"
        "    model: M2\n"
        "```\n"
    )
    prompts = load_prompts(md)
    assert len(prompts) == 1
    assert prompts[0].brand == "B1"
