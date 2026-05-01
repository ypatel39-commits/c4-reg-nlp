"""Tests for embedding shape, top-K matching, and gap detection.

Uses synthetic vectors so the heavy sentence-transformers model is NOT
required to run these tests.
"""
import numpy as np

from c4_reg_nlp.match import (
    DEFAULT_TOP_K,
    ClauseResult,
    gaps,
    match_clauses,
)
from c4_reg_nlp.regtext import Clause
from c4_reg_nlp.taxonomy import Control


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


def _make_controls(n=5, dim=4):
    out = []
    for i in range(n):
        v = np.zeros(dim, dtype=np.float32)
        v[i % dim] = 1.0
        out.append(
            Control(
                id=f"X-{i}",
                framework="NIST",
                family_or_component="Test",
                name=f"Control {i}",
                description=f"Synthetic control {i}",
            )
        )
    return out


def test_match_top_k_logic():
    dim = 4
    controls = _make_controls(n=4, dim=dim)
    control_vecs = np.eye(dim, dtype=np.float32)
    # Clause vec aligned exactly with control 2.
    clause_vec = _unit(np.array([0.0, 0.0, 1.0, 0.0]))
    clauses = [Clause(clause_id="C001", text="dummy", section="S1")]
    results = match_clauses(
        clauses,
        controls,
        top_k=2,
        gap_threshold=0.1,
        clause_vecs=clause_vec.reshape(1, -1),
        control_vecs=control_vecs,
    )
    assert len(results) == 1
    r = results[0]
    assert len(r.matches) == 2
    assert r.matches[0].control_id == "X-2"
    assert r.matches[0].similarity == 1.0
    assert r.is_gap is False


def test_gap_detection_when_low_similarity():
    dim = 4
    controls = _make_controls(n=3, dim=dim)
    control_vecs = np.eye(3, dim, dtype=np.float32)
    # Clause vec orthogonal to all controls -> all sims == 0 -> gap.
    clause_vec = _unit(np.array([0.0, 0.0, 0.0, 1.0]))
    clauses = [Clause(clause_id="C001", text="orthogonal", section=None)]
    results = match_clauses(
        clauses,
        controls,
        top_k=DEFAULT_TOP_K,
        gap_threshold=0.5,
        clause_vecs=clause_vec.reshape(1, -1),
        control_vecs=control_vecs,
    )
    assert results[0].is_gap is True
    assert gaps(results) == results


def test_clause_result_to_row_shape():
    cr = ClauseResult(
        clause_id="C001",
        section="Section 1",
        text="hello",
        matches=[],
        is_gap=True,
    )
    row = cr.to_row()
    assert row["clause_id"] == "C001"
    assert row["is_gap"] is True
    assert row["best_similarity"] == 0.0


def test_match_handles_top_k_larger_than_controls():
    dim = 3
    controls = _make_controls(n=2, dim=dim)
    control_vecs = np.eye(2, dim, dtype=np.float32)
    clause_vec = _unit(np.array([1.0, 0.0, 0.0]))
    clauses = [Clause(clause_id="C001", text="x", section=None)]
    results = match_clauses(
        clauses,
        controls,
        top_k=10,
        gap_threshold=0.1,
        clause_vecs=clause_vec.reshape(1, -1),
        control_vecs=control_vecs,
    )
    # Only 2 controls -> at most 2 matches even if top_k=10.
    assert len(results[0].matches) == 2
