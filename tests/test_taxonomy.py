"""Tests for taxonomy loader."""
from c4_reg_nlp.taxonomy import load_taxonomy, taxonomy_summary


def test_load_taxonomy_counts():
    controls = load_taxonomy()
    summary = taxonomy_summary(controls)
    assert summary["NIST"] == 30
    assert summary["COSO"] == 10
    assert len(controls) == 40


def test_control_to_text_nonempty():
    controls = load_taxonomy()
    for c in controls:
        text = c.to_text()
        assert c.id in text
        assert len(text) > 30


def test_control_ids_unique():
    controls = load_taxonomy()
    ids = [c.id for c in controls]
    assert len(ids) == len(set(ids))
