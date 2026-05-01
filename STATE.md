# STATE — c4-reg-nlp

Last updated: 2026-04-30

## Status

**Phase 1 complete (MVP).** End-to-end pipeline runs offline with no API keys.

## What works

- `data/controls.json` — 30 NIST 800-53 rev 5 controls (AC, AU, IA, RA, SC, SI) + 10 COSO 2013 principles.
- `data/sample_rule.txt` — 6-section excerpt of SEC final rule 33-11275 (climate disclosures, March 2024). Used to keep tests offline.
- `src/c4_reg_nlp/taxonomy.py` — JSON loader, dataclass model, `to_text()` for embedding.
- `src/c4_reg_nlp/regtext.py` — local + URL loader, paragraph/section segmenter.
- `src/c4_reg_nlp/embed.py` — `sentence-transformers/all-MiniLM-L6-v2` + chromadb persistent store at `data/chroma/`.
- `src/c4_reg_nlp/match.py` — cosine top-K + gap detection (default threshold 0.35).
- `app.py` — Streamlit UI (paste rule -> table of mappings + gaps panel).
- `scripts/run_demo.py` — CLI that writes `docs/sample-mapping.csv` + `docs/gaps.json`.
- `tests/` — 5 pytest modules: smoke, taxonomy, regtext, embed (skipped if ST not installed), match (synthetic vectors, no model required).

## Design decisions

- **Single embedding model.** `all-MiniLM-L6-v2` only. Small, fast, MIT, local download.
- **Cosine similarity in chromadb.** `metadata={"hnsw:space": "cosine"}` on the controls collection.
- **Match math runs on numpy.** ChromaDB is used for persistence; the actual top-K computation in `match.py` uses a numpy cosine matrix because vectors are L2-normalized at embedding time.
- **Gap threshold 0.40.** Tuned on the sample rule (best-match similarities ranged 0.35-0.53) so that climate-specific clauses (e.g., GHG emissions disclosure, financial statement effects of severe weather) land in the gap list while clauses about access/audit/governance map cleanly to existing NIST/COSO controls.
- **Random seed 42** in `embed.py::_seed_everything`.
- **All files <= 300 lines.**

## Known limitations / future work

- Single rule sample. Phase 2 should load the live SEC HTML and CFTC proposed rule index.
- No clause-level legal entity recognition (would need spaCy or a law-tuned model).
- No human-in-the-loop annotation workflow.
- No domain-tuned embedding (e.g., legal-bert) — adding one is the natural Phase 3.

## How to extend

1. Add controls -> edit `data/controls.json`, re-run `scripts/run_demo.py`.
2. Map a different rule -> drop a `.txt` in `data/` and pass `--rule-path`.
3. Swap the embedder -> change `MODEL_NAME` in `embed.py`. Tests pin the dim to 384 so re-tune if you go bigger.
