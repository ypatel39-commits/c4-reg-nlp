"""Lightweight tests for embed module. Skipped if sentence-transformers unavailable."""
import importlib.util

import numpy as np
import pytest

from c4_reg_nlp.embed import MODEL_NAME, embed_texts

HAS_ST = importlib.util.find_spec("sentence_transformers") is not None


@pytest.mark.skipif(not HAS_ST, reason="sentence-transformers not installed")
def test_embed_shape():
    texts = ["access control policy", "audit logging"]
    vecs = embed_texts(texts)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == 2
    # all-MiniLM-L6-v2 produces 384-dim vectors.
    assert vecs.shape[1] == 384
    # Vectors are L2-normalized.
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_model_name_constant():
    assert MODEL_NAME == "all-MiniLM-L6-v2"


def test_embed_empty_returns_zero_array():
    vecs = embed_texts([])
    assert vecs.shape == (0, 384)
