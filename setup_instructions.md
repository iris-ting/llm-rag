# Setup instruction

**Purpose.** This document explains how to set up and run each step of the assignment—**from dataset setup to advanced evaluation**—using the scripts in this repo. Follow Steps **1–6**; each step states its **deliverable**, what it **does**, the **command to run**, and **where the outputs go / how they’re used**.

---

## Step 1: **Domain Documents**
**Deliverable:** Dataset Setup

**What it does.** Creates the environment, installs dependencies, downloads/caches **RAG Mini Wikipedia**, and builds a first FAISS index.

```bash
# create & activate (Python 3.11 recommended)
conda create -n llm-rag python=3.11 -y
conda activate llm-rag
pip install -U pip
pip install -r requirements.txt
```
---

## Step 2: **Build Naive RAG**
**Deliverable:** Functional RAG Pipeline

**What it does.** Implements baseline retrieval (**sentence-transformers** `all-MiniLM-L6-v2` + FAISS) and generation; runs a simple internal eval.

```bash
# dataset + index smoke test
python -m src.naive_rag
# run baseline evaluation (top‑1, instruction prompt)
python -m src.evaluation
```

**Outputs & use.** Cached dataset/index (local cache). Baseline results saved under `results/` (e.g., `results/para_results/naive_results.json`). These are the **reference numbers** for later comparisons in the Step‑2 write‑up.

---

## Step 3: **Evaluation Phase I**
**Deliverable:** Initial Evaluation Results

**What it does.** Compares **prompting strategies** (e.g., *Instruction* and optionally *Persona*) with **top‑1** context; reports **EM/F1**.

```bash
# instruction prompt, top‑1 (default in your script)
python -m src.evaluation

# (optional) if PROMPTS["persona"] is wired:
# python -m src.evaluation --prompt persona
```

**Outputs & use.** Per‑prompt baseline metrics (JSON/printouts). Use them in the analysis to argue which prompt works best.

---

## Step 4: **Experimentation**
**Deliverable:** Parameter Comparison Analysis

**What it does.** Sweeps **embedding models/sizes** and **retrieval variations** (`top_k ∈ {3,5,10}`; strategies concat/MMR) with a minimal runner.

```bash
python -m src.para_experiments
```

**Outputs & use.** Written to **`results/para_runs/para_results/`**:
- Per‑run JSON: `run_<prompt>_k{3|5|10}_{concat|mmr}_{model}.json`
- `comparison_analysis.csv` — consolidated table
- `naive_results.json`, `enhanced_results.json` — summaries per setting

Use these tables to justify defaults—e.g., analyze **3 prompts × top‑K {3,5,10}** across ≥2 embeddings—and pick the combo you’ll carry into Step 6.

---

## Step 5: **Add Two Advanced RAG Features**
**Deliverable:** Enhanced RAG Implementation

**What it does.** Adds at least **two** features (in this repo: **HyDE** query rewriting, **MMR** selection, optional **cross‑encoder reranker**) and integrates them with the pipeline.

```bash
# run the enhanced pipeline (defaults)
python -m src.enhanced_rag

# optional flags
python -m src.enhanced_rag \
  --limit 100 \
  --topk 5 \
  --strategy mmr \        # mmr | concat
  --hyde \                # enable HyDE
  --reranker \            # enable cross-encoder reranker (if installed)
  --prompt instruction
  ```

**Outputs & use.** The enhanced behavior is validated in Step 6.

---

## Step 6: **Advanced Evaluation with RAGAs or ARES**
**Deliverable:** Automated Evaluation Report

**What it does.** Uses **RAGAS** to score **naive vs. enhanced** with **Faithfulness, Context Precision, Context Recall, Answer Helpfulness/Relevancy**.

```bash
# one‑time: enable judge LLM (OpenAI)
conda env config vars set OPENAI_API_KEY='sk-...'
conda deactivate && conda activate llm-rag
export RAGAS_MAX_WORKERS=1
export TOKENIZERS_PARALLELISM=false

# run RAGAS (current script writes naive & enhanced summaries)
python -m src.ragas_eval --limit 100 --topk 5 --prompt instruction
```

**Outputs & use.** Written to **`results/ragas_eval/`**:
- `naive_items_instruction.csv`, `summary_naive_instruction.json` — per‑item & mean metrics for **naive**
- `enhanced_items_instruction.csv`, `summary_enhanced_instruction.json` — per‑item & mean metrics for **enhanced**
- **`summary_compare.csv`** and `summary_compare_instruction.csv` — **side‑by‑side means** used directly in the Step‑6 report  
  *Tip:* Repeat with other prompts or top‑K to produce **3 prompts × top‑K=5** comparisons required by the assignment.
