"""Load and represent the internal control taxonomy (NIST 800-53 + COSO 2013)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "data" / "controls.json"


@dataclass(frozen=True)
class Control:
    """A single internal control entry from one of the supported frameworks."""

    id: str
    framework: str  # "NIST" or "COSO"
    family_or_component: str
    name: str
    description: str

    def to_text(self) -> str:
        """Concatenate fields to a single text blob suitable for embedding."""
        return f"{self.id} {self.name}: {self.description}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "framework": self.framework,
            "family_or_component": self.family_or_component,
            "name": self.name,
            "description": self.description,
        }


def load_taxonomy(path: Path | str | None = None) -> list[Control]:
    """Read controls.json and return a flat list of Control objects."""
    p = Path(path) if path else DEFAULT_TAXONOMY_PATH
    if not p.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    controls: list[Control] = []
    for item in raw.get("nist_800_53_rev5", []):
        controls.append(
            Control(
                id=item["id"],
                framework="NIST",
                family_or_component=item["family"],
                name=item["name"],
                description=item["description"],
            )
        )
    for item in raw.get("coso_2013", []):
        controls.append(
            Control(
                id=item["id"],
                framework="COSO",
                family_or_component=item["component"],
                name=item["principle"],
                description=item["description"],
            )
        )
    return controls


def iter_controls(path: Path | str | None = None) -> Iterator[Control]:
    yield from load_taxonomy(path)


def taxonomy_summary(controls: list[Control]) -> dict[str, int]:
    """Return counts per framework — useful for sanity checks."""
    out: dict[str, int] = {}
    for c in controls:
        out[c.framework] = out.get(c.framework, 0) + 1
    return out
