"""Embed controls and rule clauses with sentence-transformers; persist via chromadb."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np

from .taxonomy import Control

# Lazy imports of heavy deps inside functions so unit tests can stub things.

MODEL_NAME = "all-MiniLM-L6-v2"
RANDOM_STATE = 42
DEFAULT_CHROMA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "chroma"
)
CONTROLS_COLLECTION = "controls"


def _seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def get_model(model_name: str = MODEL_NAME):
    """Return a loaded SentenceTransformer model."""
    _seed_everything()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: Iterable[str], model=None) -> np.ndarray:
    """Encode a batch of texts to a (N, D) float32 numpy array."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    if model is None:
        model = get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)


def get_chroma_client(persist_path: Path | str | None = None):
    """Return a chromadb PersistentClient anchored at ``persist_path``."""
    import chromadb

    p = Path(persist_path) if persist_path else DEFAULT_CHROMA_PATH
    p.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(p))


def index_controls(
    controls: list[Control],
    *,
    persist_path: Path | str | None = None,
    model=None,
):
    """Embed controls and write/replace the chromadb collection."""
    client = get_chroma_client(persist_path)
    try:
        client.delete_collection(CONTROLS_COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(
        name=CONTROLS_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    texts = [c.to_text() for c in controls]
    vecs = embed_texts(texts, model=model)
    coll.add(
        ids=[c.id for c in controls],
        embeddings=vecs.tolist(),
        documents=texts,
        metadatas=[
            {
                "framework": c.framework,
                "family_or_component": c.family_or_component,
                "name": c.name,
            }
            for c in controls
        ],
    )
    return coll


def get_controls_collection(persist_path: Path | str | None = None):
    client = get_chroma_client(persist_path)
    return client.get_collection(CONTROLS_COLLECTION)
