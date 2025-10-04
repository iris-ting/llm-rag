# Phase 3 — Enhanced RAG System

**Deliverable:** Enhanced RAG implementation (**code + analysis**).  
This document describes our enhancements (HyDE + MMR + optional reranker), how to run them, and the **RAGAS-based comparison** against the naive baseline.

---

## Enhancements Implemented
- **HyDE query rewriting**: generate a short hypothetical answer, append to the query, embed, and retrieve — improves semantic recall.
- **MMR selection**: greedy diversity selection to reduce duplicate contexts and keep complementary evidence.
- **(Optional) Cross-encoder reranker**: re-orders candidates using a pairwise model when enabled.

---

## How to Run
```bash
# enhanced pipeline (recommended settings)
python -m src.enhanced_rag --limit 100 --topk 10 --strategy mmr --hyde --prompt instruction
# add --reranker if the CrossEncoder is available
```

Outputs: `results/enhanced_results.json` (enhanced) and, from parameter sweeps, `results/para_runs/para_results/…`.  
For RAGAS, we used `python -m src.ragas_eval --limit 100 --topk 5 --prompt instruction` and then compared **naive** vs **enhanced** in `results/ragas_eval/summary_compare.csv`.

---

## RAGAS Metrics (mean)

| Mode      | Faithfulness | Context Precision | Context Recall | Answer Relevancy |
|-----------|--------------|-------------------|----------------|------------------|
| enhanced  | 0.9167 | 0.8361        | 0.8875     | 0.8488       |
| naive     | 0.8551 | 0.7109        | 0.8159     | 0.8225       |
| **Δ (abs.)** | **+0.0616** | **+0.1252**       | **+0.0716**    | **+0.0263**      |
| **Δ (rel.)** | **+7.20%** | **+17.61%**      | **+8.78%**   | **+3.20%**     |

**Interpretation**
- **Faithfulness**: +0.0616 (≈7.2% relative). Answers cite and align with contexts more reliably after HyDE+MMR.
- **Context precision**: +0.1252 (≈17.6%). MMR reduces redundant/irrelevant passages; optional reranking sharpens ordering.
- **Context recall**: +0.0716 (≈8.8%). HyDE tends to pull in semantically related passages that naive lexical-seeming queries miss.
- **Answer relevancy**: +0.0263 (≈3.2%). Gains are smaller but consistent; precision improvements appear to translate to slightly better answers.

---

## Analysis
1. **HyDE → better retrieval seeds**: rewriting the query with a hypothesized answer often injects missing entities/relations, improving candidate coverage.
2. **MMR → diverse evidence**: avoids stacking near-duplicates, which increases **context precision** without sacrificing recall.
3. **Reranker (when enabled)**: helps especially when `topk` is large, by promoting passages with stronger pairwise relevance.

---

## Notes & Trade-offs
- Latency increases with HyDE and especially with a cross-encoder reranker. If speed is critical, keep **MMR on** and disable `--reranker`.
- From Phase 2 grid, **384‑d + top‑K=10** was a sweet spot; enhanced runs reuse these defaults.