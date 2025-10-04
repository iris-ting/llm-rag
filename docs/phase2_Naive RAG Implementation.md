# Naive RAG Implementation-Parameter Comparison (Step 4/ Phase 2)

**Scope.** This document summarizes the baseline system and reports both the **initial evaluation** (Step 3) and the **parameter comparison** (Step 4).

---

## 1) System Overview (minimal, modular)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384‑d) via `utils.build_faiss_index(passages)`  
- **Vector DB:** FAISS (IndexIP) stored in‑memory (rebuilt quickly from cache)  
- **Retrieval:** `utils.search_top_k(query, index, embed_model, passages, k)` → returns top‑K passages  
- **Generation:** `evaluation.load_generator()` (Hugging Face pipeline). Prompt template lives in `evaluation.PROMPTS[...]`  
- **Post‑processing:** `evaluation.postprocess_prediction()` normalizes the text before scoring

**Entry points**
- `src/naive_rag.py` — smoke test: load data, build index, run a tiny retrieval/generation demo  
- `src/evaluation.py` — baseline scoring runner (EM/F1), configurable `prompt_style` and `top_k`

**How to run**
```bash
# build index + quick demo
python -m src.naive_rag

# baseline evaluation (default: instruction prompt, top-1)
python -m src.evaluation
```

---

## 2) Step 3 — Initial Evaluation (top‑1, prompt comparison)
**Task.** Pass the **top‑1** retrieved passage to the prompt; compare **prompting strategies**; compute **EM/F1** (SQuAD v1 metric via `evaluate`).

**Prompts compared**
- **Instruction** (default)
- **Persona** (*optional*, if `PROMPTS["persona"]` is defined in your local checkout)

**Command**
```bash
# Instruction, top-1
python -m src.evaluation

# Persona, top-1 (if available)
# python -m src.evaluation --prompt persona
```

**Baseline result (our run, Instruction, top‑1)**  
EM **34.0**, F1 **38.34**

**Finding.** Instruction was the most stable for top‑1; persona did not consistently improve EM/F1 in our quick checks. We **carry Instruction forward** into the experiments.

---

## 3) Step 4 — Parameter Comparison (embeddings × selection × top‑K)
**Grid.** Two embedding sizes and two selection strategies, evaluated at multiple top‑K values on 100 samples:
- **Embeddings:** 384‑d (`all-MiniLM-L6-v2`) vs. 512‑d (`distiluse-base-multilingual-cased-v2`)
- **Selection:** `concat` vs. `mmr`
- **Top‑K:** {3, 5, 10}

**Runner**
```bash
python -m src.para_experiments
```

**Best Overall**
- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` (dim 384)  
- **Selection**: `concat`, **Top‑K**: 10  
- **Scores**: **F1 60.92**, EM 53.00  
- **Speed**: 1.687 sec/question

**Best by Strategy**
- `concat` → `all-MiniLM-L6-v2` (384), Top‑K 10: F1 60.92, EM 53.00  
- `mmr`   → `all-MiniLM-L6-v2` (384), Top‑K 10: F1 59.95, EM 53.00

**Top Results (sorted by F1)**

| Embedding                                                  |   Dim |   Top‑K | Strategy   |   EM |    F1 |   Sec/Q |
|:-----------------------------------------------------------|------:|--------:|:-----------|-----:|------:|--------:|
| sentence-transformers/all-MiniLM-L6-v2                     |   384 |      10 | concat     |   53 | 60.92 |   1.687 |
| sentence-transformers/all-MiniLM-L6-v2                     |   384 |      10 | mmr        |   53 | 59.95 |   1.532 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 |   512 |      10 | concat     |   49 | 56.59 |   2.177 |
| sentence-transformers/all-MiniLM-L6-v2                     |   384 |       5 | concat     |   48 | 56.13 |   0.774 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 |   512 |      10 | mmr        |   48 | 55.69 |   2.028 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 |   512 |       5 | mmr        |   47 | 54.05 |   0.896 |
| sentence-transformers/all-MiniLM-L6-v2                     |   384 |       3 | concat     |   45 | 52.85 |   0.394 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 |   512 |       5 | concat     |   44 | 51.37 |   1.025 |
| sentence-transformers/all-MiniLM-L6-v2                     |   384 |       5 | mmr        |   44 | 50.72 |   0.714 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 |   512 |       3 | concat     |   44 | 50.18 |   0.541 |

**Brief analysis**
- Increasing **Top‑K** improved recall/F1; `mmr` reduced redundancy and was close to or better than `concat` at smaller K.  
- The best 512‑d run scored **≈4.3 F1** lower than the best 384‑d run; the lighter model is also faster (Sec/Q).  
- **Recommendation:** use **384‑d + Top‑K 10**; start with `concat` for simplicity, switch to `mmr` if you see redundancy.

**Artifacts**
- Per‑run JSONs and consolidated table at `results/para_runs/para_results/`  
  - `run_<prompt>_k{3|5|10}_{concat|mmr}_{model}.json`  
  - `comparison_analysis.csv`  
  - `naive_results.json`, `enhanced_results.json`
