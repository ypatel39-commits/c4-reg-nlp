"""End-to-end demo: load taxonomy + sample rule, embed, match, write artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from c4_reg_nlp.embed import get_model, index_controls
from c4_reg_nlp.match import DEFAULT_GAP_THRESHOLD, DEFAULT_TOP_K, gaps, match_clauses
from c4_reg_nlp.regtext import load_local, segment
from c4_reg_nlp.taxonomy import load_taxonomy, taxonomy_summary

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


@click.command()
@click.option("--rule-path", type=click.Path(exists=True, path_type=Path),
              default=ROOT / "data" / "sample_rule.txt", show_default=True)
@click.option("--top-k", type=int, default=DEFAULT_TOP_K, show_default=True)
@click.option("--threshold", type=float, default=DEFAULT_GAP_THRESHOLD,
              show_default=True)
@click.option("--persist/--no-persist", default=True,
              help="Persist control embeddings to chromadb.")
def main(rule_path: Path, top_k: int, threshold: float, persist: bool) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)

    click.echo("[1/5] Loading taxonomy + rule text...")
    controls = load_taxonomy()
    click.echo(f"      taxonomy summary: {taxonomy_summary(controls)}")
    text = load_local(rule_path)
    clauses = segment(text)
    click.echo(f"      segmented into {len(clauses)} clauses")

    click.echo("[2/5] Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    model = get_model()

    if persist:
        click.echo("[3/5] Indexing controls into chromadb...")
        index_controls(controls, model=model)
    else:
        click.echo("[3/5] Skipping chromadb persistence (--no-persist)")

    click.echo("[4/5] Matching clauses to controls...")
    results = match_clauses(
        clauses, controls, top_k=top_k, gap_threshold=threshold, model=model
    )
    gap_results = gaps(results)

    click.echo("[5/5] Writing artifacts...")
    df = pd.DataFrame([r.to_row() for r in results])
    csv_path = DOCS / "sample-mapping.csv"
    df.to_csv(csv_path, index=False)

    gap_path = DOCS / "gaps.json"
    gap_payload = {
        "rule_path": str(rule_path),
        "top_k": top_k,
        "threshold": threshold,
        "n_clauses": len(results),
        "n_gaps": len(gap_results),
        "gaps": [
            {
                "clause_id": g.clause_id,
                "section": g.section,
                "best_similarity": round(g.best_similarity(), 4),
                "text": g.text,
                "top_matches": [
                    {"id": m.control_id, "name": m.name,
                     "similarity": round(m.similarity, 4)}
                    for m in g.matches
                ],
            }
            for g in gap_results
        ],
    }
    gap_path.write_text(json.dumps(gap_payload, indent=2), encoding="utf-8")

    click.echo(f"      wrote {csv_path}")
    click.echo(f"      wrote {gap_path}")
    click.echo(
        f"DONE: {len(results)} clauses mapped; {len(gap_results)} gaps flagged."
    )


if __name__ == "__main__":
    main()
