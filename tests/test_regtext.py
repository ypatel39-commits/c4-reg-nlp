"""Tests for rule text loading + segmentation."""
from c4_reg_nlp.regtext import load_local, segment


def test_load_sample_rule():
    text = load_local()
    assert "Climate" in text or "climate" in text
    assert len(text) > 500


def test_segment_produces_clauses():
    text = load_local()
    clauses = segment(text)
    # Sample rule has 6 sections + intro paragraph; expect at least 5 clauses.
    assert len(clauses) >= 5
    assert all(c.text for c in clauses)
    assert all(c.clause_id.startswith("C") for c in clauses)


def test_segment_assigns_sections():
    text = load_local()
    clauses = segment(text)
    sectioned = [c for c in clauses if c.section]
    assert sectioned, "Expected at least some clauses to capture a Section heading"
