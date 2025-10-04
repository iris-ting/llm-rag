# Technical Appendix
## A. Reproducibility Checklist

**Environment (Python 3.11 recommended)**
```bash
conda create -n llm-rag python=3.11 -y
conda activate llm-rag
pip install -U pip
pip install -r requirements.txt
```

**RAGAS judge (OpenAI)**
```bash
# bind once to the conda env
conda env config vars set OPENAI_API_KEY='sk-...'
conda deactivate && conda activate llm-rag

# stability during judging
export RAGAS_MAX_WORKERS=1
export TOKENIZERS_PARALLELISM=false
```

**Seeds & determinism**
- Use fixed `random.seed` / `numpy.random.seed` where supported by your scripts.
- Embeddings from Sentence-Transformers are deterministic given model weights; generation may still vary slightly depending on decoding settings.

---

## B. Data & EDA

**Dataset**: *RAG Mini Wikipedia* on Hugging Face.  
**Configs**: `text-corpus` (passages) and `question-answer` (eval).  
**Loader**: `src/utils.py::robust_load_miniwiki()` (cache-first; falls back to explicit configs).

**EDA & Deduplication**
- After loading, I print passage/test counts and check textual duplicates.
- **I found 4 duplicated passages and removed them** using `inspect_and_deduplicate()`.
- Columns used: passages → `text` (and optional `title`); QA → `question`, `answer`.

---

## C. Project Structure (relevant paths)

```
src/
  utils.py
  naive_rag.py
  evaluation.py
  para_experiments.py
  enhanced_rag.py
  ragas_eval.py
results/
  naive_results.json
  enhanced_results.json
  para_runs/para_results/
    comparison_analysis.csv
    run_<prompt>_k{3|5|10}_{concat|mmr}_{model}.json
  ragas_eval/
    naive_items_<prompt>.csv
    enhanced_items_<prompt>.csv
    summary_naive_<prompt>.json
    summary_enhanced_<prompt>.json
    summary_compare.csv
```

---

## D. Configuration Defaults

| Component | Default | Alternatives / Notes |
|---|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384‑d) | `distiluse-base-multilingual-cased-v2` (512‑d) |
| Vector DB | FAISS (IndexIP) | In‑memory; cached for speed |
| Retrieval depth (`topk`) | **10** | 3, 5 used in sweeps |
| Selection | **MMR (λ=0.5)** | `concat` (fastest; may duplicate) |
| Query rewrite | **HyDE: on** | off for ablation |
| Reranker | off by default | CrossEncoder (enable for precision) |
| Prompt style | **Instruction** | Persona (optional) |

**CLI examples**
```bash
# baseline EM/F1
python -m src.evaluation --prompt instruction

# parameter sweeps
python -m src.para_experiments

# enhanced pipeline (recommended)
python -m src.enhanced_rag --limit 100 --topk 10 --strategy mmr --hyde --prompt instruction

# RAGAS comparison (naive vs enhanced)
python -m src.ragas_eval --limit 5 --topk 5 --prompt instruction
```

---

## E. Prompt Template (Instruction; schematic)

```
You are a helpful assistant. Given the context, answer the question concisely.
Context:
{ctx}

Question:
{q}

Answer:
```

The exact string lives in `evaluation.PROMPTS["instruction"]`; placeholders are `{q}` and `{ctx}`. Persona (if used) adds a role description ahead of the same structure.

---


## F. Commands to Reproduce Key Artifacts

```bash
# 1) dataset + index smoke test
python -m src.naive_rag

# 2) baseline EM/F1 (naive)
python -m src.evaluation --prompt instruction

# 3) parameter sweeps (embedding × strategy × K)
python -m src.para_experiments

# 4) enhanced pipeline (HyDE + MMR; reranker optional)
python -m src.enhanced_rag --limit 100 --topk 10 --strategy mmr --hyde --prompt instruction

# 5) RAGAS (naive vs enhanced under same settings)
python -m src.ragas_eval --limit 5 --topk 5 --prompt instruction
```

---

## G. Troubleshooting

- **Hugging Face dataset config error**: If `rag-datasets/rag-mini-wikipedia` complains about a missing config, call with explicit configs (`text-corpus`, `question-answer`) or ensure the cache has them. `robust_load_miniwiki()` already handles this path.

- **`clean_up_tokenization_spaces` warning**: Harmless; upcoming Transformers default change. Safe to ignore.

- **Sequence length > model max**: You may see `Token indices sequence length ...`; inputs are truncated by the pipeline—safe for this assignment.

- **OpenAI key not found**: Ensure you set `OPENAI_API_KEY` in the **same** environment used to run the script (IDE terminals may not inherit conda vars).

- **429 / timeout during RAGAS**: Lower concurrency (`RAGAS_MAX_WORKERS=1`), reduce `--limit`, and retry. Consider `--topk 5` for faster loops.

- **LangChain imports**: Not required. This repo uses direct HF pipelines and RAGAS; drop `LangchainLLM` imports if seen in external snippets.
