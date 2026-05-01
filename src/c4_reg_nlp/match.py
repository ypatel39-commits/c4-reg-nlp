"""Match rule clauses to controls and flag gaps."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .embed import embed_texts
from .regtext import Clause
from .taxonomy import Control

DEFAULT_TOP_K = 3
# Cosine similarity below this -> potential gap.
# Tuned on the bundled SEC climate-disclosure sample so that climate-specific
# clauses (e.g. GHG emissions, financial statement effects of severe weather)
# surface as gaps, while access/audit/governance clauses map cleanly.
DEFAULT_GAP_THRESHOLD = 0.40


@dataclass
class Match:
    """One control-match for a clause, with similarity."""

    control_id: str
    framework: str
    name: str
    similarity: float


@dataclass
class ClauseResult:
    """Per-clause mapping output."""

    clause_id: str
    section: str | None
    text: str
    matches: list[Match] = field(default_factory=list)
    is_gap: bool = False

    def best_similarity(self) -> float:
        return max((m.similarity for m in self.matches), default=0.0)

    def to_row(self) -> dict:
        top = self.matches[0] if self.matches else None
        return {
            "clause_id": self.clause_id,
            "section": self.section or "",
            "is_gap": self.is_gap,
            "best_control_id": top.control_id if top else "",
            "best_control_name": top.name if top else "",
            "best_framework": top.framework if top else "",
            "best_similarity": round(self.best_similarity(), 4),
            "all_matches": "; ".join(
                f"{m.control_id}({m.similarity:.2f})" for m in self.matches
            ),
            "clause_text": self.text,
        }


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between (N,D) and (M,D); inputs assumed L2-normalized."""
    return a @ b.T


def match_clauses(
    clauses: list[Clause],
    controls: list[Control],
    *,
    top_k: int = DEFAULT_TOP_K,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    model=None,
    clause_vecs: np.ndarray | None = None,
    control_vecs: np.ndarray | None = None,
) -> list[ClauseResult]:
    """For each clause, find top-K controls by cosine similarity; flag gaps.

    Pre-computed ``clause_vecs`` / ``control_vecs`` may be passed to skip
    re-embedding (used by tests).
    """
    if not clauses:
        return []
    if not controls:
        raise ValueError("controls list is empty")
    if control_vecs is None:
        control_vecs = embed_texts([c.to_text() for c in controls], model=model)
    if clause_vecs is None:
        clause_vecs = embed_texts([c.text for c in clauses], model=model)

    sims = _cosine(clause_vecs, control_vecs)  # (n_clauses, n_controls)
    k = min(top_k, len(controls))
    results: list[ClauseResult] = []
    for i, clause in enumerate(clauses):
        row = sims[i]
        top_idx = np.argsort(-row)[:k]
        matches = [
            Match(
                control_id=controls[j].id,
                framework=controls[j].framework,
                name=controls[j].name,
                similarity=float(row[j]),
            )
            for j in top_idx
        ]
        best = matches[0].similarity if matches else 0.0
        results.append(
            ClauseResult(
                clause_id=clause.clause_id,
                section=clause.section,
                text=clause.text,
                matches=matches,
                is_gap=best < gap_threshold,
            )
        )
    return results


def gaps(results: Iterable[ClauseResult]) -> list[ClauseResult]:
    return [r for r in results if r.is_gap]
