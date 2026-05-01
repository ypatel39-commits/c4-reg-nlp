"""Streamlit UI: paste rule text -> see clause-by-clause control mapping + gaps."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from c4_reg_nlp.embed import get_model
from c4_reg_nlp.match import DEFAULT_GAP_THRESHOLD, DEFAULT_TOP_K, gaps, match_clauses
from c4_reg_nlp.regtext import load_local, segment
from c4_reg_nlp.taxonomy import load_taxonomy

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "sample_rule.txt"


@st.cache_resource
def cached_model():
    return get_model()


@st.cache_data
def cached_taxonomy():
    return load_taxonomy()


def main() -> None:
    st.set_page_config(page_title="C4 Reg-NLP", layout="wide")
    st.title("C4 Reg-NLP — Rule -> Control Mapper")
    st.caption(
        "Maps proposed SEC/CFTC regulatory rule text to NIST 800-53 + COSO 2013 "
        "controls via sentence-transformer embeddings; flags clauses with no "
        "strong match as potential gaps."
    )

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top-K controls", 1, 10, DEFAULT_TOP_K)
        threshold = st.slider(
            "Gap threshold (cosine sim)", 0.0, 1.0, DEFAULT_GAP_THRESHOLD, 0.05
        )
        use_sample = st.checkbox("Use bundled sample rule", value=True)

    if use_sample:
        default_text = load_local(SAMPLE_PATH)
    else:
        default_text = ""
    rule_text = st.text_area("Rule text", value=default_text, height=300)

    if not st.button("Run mapping"):
        st.info("Paste rule text and click Run mapping.")
        return

    if not rule_text.strip():
        st.error("No rule text provided.")
        return

    with st.spinner("Loading taxonomy + model..."):
        controls = cached_taxonomy()
        model = cached_model()
    clauses = segment(rule_text)
    st.write(f"Segmented into **{len(clauses)}** clauses; "
             f"taxonomy has **{len(controls)}** controls.")

    with st.spinner("Embedding + matching..."):
        results = match_clauses(
            clauses, controls, top_k=top_k, gap_threshold=threshold, model=model
        )

    rows = [r.to_row() for r in results]
    df = pd.DataFrame(rows)
    st.subheader("Clause -> Control mapping")
    st.dataframe(df, use_container_width=True)

    gap_rows = [r.to_row() for r in gaps(results)]
    st.subheader(f"Flagged gaps ({len(gap_rows)})")
    if gap_rows:
        st.dataframe(pd.DataFrame(gap_rows), use_container_width=True)
    else:
        st.success("No gaps detected at the configured threshold.")


if __name__ == "__main__":
    main()
