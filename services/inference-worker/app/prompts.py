"""Loader for the MobileCLIP zero-shot prompts file (DESIGN.md §5.4).

The canonical prompts file is `docs/prompts.md` at the repo root — a Markdown
document with a single ```yaml fenced block. The loader extracts the YAML body
and validates each entry against the `Prompt` pydantic model.

Plain `.yaml` / `.yml` files are also accepted (any non-`.md` suffix is parsed
as YAML directly), which keeps test fixtures and ad-hoc overrides simple.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

# Match the first ```yaml fenced block. DOTALL so `.` spans newlines; the
# capture group is non-greedy so we stop at the first closing ``` on its own
# line.
_YAML_BLOCK_RE = re.compile(r"```yaml\s*\n(.*?)\n```", re.DOTALL)

# Strip surrounding whitespace then require at least one character. This
# rejects both `""` and `"   "` while letting normal labels through.
NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class Prompt(BaseModel):
    """A single zero-shot prompt: free-form text plus its (brand, model) label.

    Brand and model are plain strings — not a `Literal` — because DESIGN.md
    explicitly allows extending the prompts file without code changes. Empty
    or whitespace-only strings are rejected so that downstream voting can rely
    on truthy labels. The model is frozen (immutable value object) and forbids
    unknown fields so that typos like `brnad:` surface as validation errors.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: NonEmptyStr
    brand: NonEmptyStr
    model: NonEmptyStr


def load_prompts(path: Path) -> list[Prompt]:
    """Load and validate the prompts file.

    If `path` ends with `.md` (case-insensitive), the first ```yaml fenced
    block is extracted and parsed. Otherwise the file is parsed as YAML
    directly.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: on any of:
            - a `.md` file with no ```yaml block
            - YAML parse failure
            - missing top-level `prompts:` key
            - `prompts` is not a list
            - `prompts` is an empty list
            - a prompt entry fails validation (missing/empty/whitespace-only
              field, or an unknown field)
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    if p.suffix.lower() == ".md":
        match = _YAML_BLOCK_RE.search(text)
        if match is None:
            raise ValueError(f"no ```yaml fenced block found in {path}")
        body = match.group(1)
    else:
        body = text

    try:
        data = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        raise ValueError(f"failed to parse YAML from {path}: {exc}") from exc

    if not isinstance(data, dict) or "prompts" not in data:
        raise ValueError(f"missing top-level 'prompts' key in {path}")

    entries = data["prompts"]
    if not isinstance(entries, list):
        raise ValueError(f"'prompts' must be a list in {path}, got {type(entries).__name__}")

    if not entries:
        raise ValueError(f"'prompts' list is empty in {path}")

    try:
        return [Prompt.model_validate(entry) for entry in entries]
    except ValidationError as exc:
        raise ValueError(f"invalid prompt entry in {path}: {exc}") from exc
