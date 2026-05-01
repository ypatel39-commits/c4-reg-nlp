"""Load and segment proposed regulatory rule text into clauses."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DEFAULT_SAMPLE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "sample_rule.txt"
)


@dataclass(frozen=True)
class Clause:
    """A single segmented clause from a rule document."""

    clause_id: str
    text: str
    section: str | None = None

    def to_dict(self) -> dict:
        return {"clause_id": self.clause_id, "section": self.section, "text": self.text}


def load_local(path: Path | str | None = None) -> str:
    """Read raw rule text from a local file."""
    p = Path(path) if path else DEFAULT_SAMPLE_PATH
    if not p.exists():
        raise FileNotFoundError(f"Rule text file not found: {p}")
    return p.read_text(encoding="utf-8")


def load_url(url: str, timeout: int = 30) -> str:
    """Fetch a public rule text or HTML page and return cleaned plain text."""
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "c4-reg-nlp/0.1"})
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "").lower()
    if "html" in ctype or url.lower().endswith((".html", ".htm")):
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    return resp.text


# Matches a leading "Section N." or "Section N:" prefix (with title up to first
# period that ends the section title). Used to peel section title off a paragraph.
_SECTION_PREFIX_RE = re.compile(
    r"^\s*(Section\s+\d+[A-Za-z]?[\.:]\s*[^.]+\.)\s*", re.IGNORECASE
)
# Matches a *line* that is purely a section heading (no body text after).
_SECTION_LINE_RE = re.compile(r"^\s*Section\s+\d+[A-Za-z]?[\.:]?\s*(.*)$", re.IGNORECASE)


def segment(text: str, min_chars: int = 80) -> list[Clause]:
    """Split rule text into clauses keyed on paragraph + section heading.

    Strategy:
      * Split on blank lines.
      * If a paragraph begins with a "Section N. <Title>." prefix, peel that
        prefix off, set it as the current section, and use the remainder as
        the clause body.
      * If a paragraph is *only* a section heading line, mark the section and
        carry it forward to the next non-heading paragraph.
      * Drop fragments shorter than ``min_chars`` to filter out noise.
    """
    clauses: list[Clause] = []
    current_section: str | None = None
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    idx = 0
    for para in paragraphs:
        body = para
        m = _SECTION_PREFIX_RE.match(para)
        if m:
            current_section = m.group(1).strip()
            body = para[m.end():].strip()
        else:
            first_line = para.splitlines()[0].strip()
            line_m = _SECTION_LINE_RE.fullmatch(first_line)
            if line_m and not line_m.group(1):
                # heading-only line
                current_section = first_line
                remainder = "\n".join(para.splitlines()[1:]).strip()
                if len(remainder) < min_chars:
                    continue
                body = remainder
        if len(body) < min_chars:
            continue
        idx += 1
        clauses.append(
            Clause(
                clause_id=f"C{idx:03d}",
                text=body,
                section=current_section,
            )
        )
    return clauses


def load_and_segment(
    source: str | Path | None = None, *, from_url: bool = False
) -> list[Clause]:
    """Convenience: load + segment in one call."""
    if from_url and isinstance(source, str):
        text = load_url(source)
    else:
        text = load_local(source)
    return segment(text)
