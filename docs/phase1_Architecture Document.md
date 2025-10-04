# Phase 1 — Domain Documents

## 1) Dataset Access
We use the **RAG Mini Wikipedia** dataset from Hugging Face:
- Passages: config **`text-corpus`** (short Wikipedia-like passages)
- QA evaluation split: config **`question-answer`**

Loader entry point: `src/utils.py::robust_load_miniwiki()` which handles cache-first loading and falls back to explicit configs if needed.

## 2) Data Understanding
After loading, we run a light sanity check:
- Print passage/test counts
- Detect and **drop duplicated passages** by text (see `inspect_and_deduplicate()`), then print counts again
- Expected columns
  - Passages: `text` (string), optionally `title`
  - QA: `question`, `answer`

Quick command (also builds the first FAISS index and ensures local cache):
```bash
python -m src.naive_rag
```

## 3) Infrastructure Planning
- **Environment:** `conda create -n llm-rag python=3.11` and `pip install -r requirements.txt`
- **CPU-first:** works on CPU; GPU is optional
- **Caching:** Hugging Face datasets cache + local FAISS index to speed up later runs
- **Folder layout (relevant):**
  - `src/` (code) — `utils.py`, `naive_rag.py`, `evaluation.py`, `para_experiments.py`, `ragas_eval.py`, `enhanced_rag.py`
  - `results/` — outputs (`para_runs/para_results`, `ragas_eval`, baseline JSONs)