# src/ragas_eval.py
"""
RAGAS evaluation (naive & enhanced) – minimal, production-ready.

Requirements (pip):
  ragas==0.1.21
  langchain==0.1.16
  langchain-community>=0.0.32,<0.1

This script evaluates two pipelines with RAGAS metrics:
  - faithfulness
  - context_precision
  - context_recall
  - answer_relevancy

Outputs (default folder: results/ragas_eval):
  - naive_items.csv / enhanced_items.csv  (per-question scores)
  - summary_naive.json / summary_enhanced.json (means)
  - summary_compare.csv (side-by-side means)
"""

from __future__ import annotations
import os, json, argparse, time
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from datasets import Dataset

# RAGAS (v0.1.21 allows passing LangChain LLM/Embeddings directly)
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy

# Local project imports
from .utils import robust_load_miniwiki, build_faiss_index, search_top_k, inspect_and_deduplicate
from .evaluation import PROMPTS, postprocess_prediction

# Local enhanced pipeline (Step 5). If missing, enhanced mode will be disabled.
try:
    from .enhanced_rag import EnhancedRAG
    HAVE_ENHANCED = True
except Exception:
    EnhancedRAG = None  # type: ignore
    HAVE_ENHANCED = False

# LangChain wrappers around local HF models
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
from langchain_community.llms import HuggingFacePipeline as LC_HF_Pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------- helpers ----------------
def build_local_llm(max_new_tokens: int = 96) -> LC_HF_Pipeline:
    tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    gen_pipe = hf_pipeline(
        "text2text-generation",
        model=mdl,
        tokenizer=tok,
        device_map="auto",
        do_sample=False,
        max_new_tokens=max_new_tokens,
        truncation=True,
    )
    return LC_HF_Pipeline(pipeline=gen_pipe)


def build_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)

def trim_to_budget(text: str, tokenizer, max_tokens: int = 512) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


# ---------------- pipelines ----------------
def run_naive(
    passages: List[str],
    questions: List[str],
    embed_model_name: str,
    top_k: int,
    prompt_style: str,
) -> Tuple[List[str], List[List[str]]]:
    """Return predictions and per-question contexts for naive pipeline."""
    index, embed_model, _ = build_faiss_index(passages, model_name=embed_model_name)

    # generator for answering (not the RAGAS judge)
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    gen = hf_pipeline("text2text-generation", model=mdl, tokenizer=tok, device_map="auto")

    preds, ctx_lists = [], []
    for q in questions:
        hits = search_top_k(q, index, embed_model, passages, k=top_k)
        ctxs = [h[2] for h in hits]
        prompt = PROMPTS[prompt_style].format(q=q, ctx="\n\n---\n\n".join(ctxs))
        prompt = trim_to_budget(prompt, tok, max_tokens=512)
        out = gen(prompt, max_new_tokens=128, truncation=True)[0]["generated_text"]
        preds.append(postprocess_prediction(out))
        ctx_lists.append(ctxs)
    return preds, ctx_lists


def run_enhanced_wrapped(
    passages: List[str],
    questions: List[str],
    top_k: int,
    prompt_style: str,
    embed_model_name: str,
    strategy: str,
    pool: int,
    use_hyde: bool,
    use_reranker: bool,
    rerank_pool: int,
) -> Tuple[List[str], List[List[str]]]:
    """
    Use your Step-5 EnhancedRAG class if available.
    """
    if not HAVE_ENHANCED:
        raise RuntimeError("EnhancedRAG class not found. Please add src/enhanced_rag.py (with EnhancedRAG).")

    rag = EnhancedRAG(
        embed_model=embed_model_name,
        gen_model="google/flan-t5-base",
        top_k_default=top_k,
        strategy=strategy,
        pool=pool,
        use_hyde=use_hyde,
        use_reranker=use_reranker,
        rerank_pool=rerank_pool,
        prompt_style=prompt_style,
    )
    rag.passages = passages[:]  # reuse already-loaded & deduped passages
    rag.build_index()

    preds, ctx_lists = [], []
    for q in questions:
        rewritten = rag.rewrite_query(q)
        ctxs = rag.retrieve(rewritten, top_k=top_k)
        pred = rag.generate(q, ctxs)
        preds.append(pred)
        ctx_lists.append(ctxs)
    return preds, ctx_lists


# ---------------- RAGAS wrapper ----------------
def ragas_score(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    golds: List[str],
    judge_llm: LC_HF_Pipeline,
    judge_emb: HuggingFaceEmbeddings,
) -> pd.DataFrame:
    """
    Build HF dataset in the format RAGAS expects and run metrics.
    Returns a DataFrame with per-question scores.
    """
    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": golds,
    })
    res = ragas_evaluate(
        dataset=ds,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_emb,
        raise_exceptions=False  # labeled NaN if error (e.g., empty context
    )

    return res.to_pandas()


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="RAGAS evaluation (naive & enhanced).")
    ap.add_argument("--limit", type=int, default=50, help="Use 50 for quick local evaluation.")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--strategy", type=str, default="mmr", choices=["concat", "mmr"])
    ap.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--prompt", type=str, default="instruction")
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--use-hyde", action="store_true", default=True)
    ap.add_argument("--use-reranker", action="store_true", default=True)
    ap.add_argument("--rerank-pool", type=int, default=50)
    ap.add_argument("--only", type=str, default="both", choices=["both", "naive", "enhanced"])
    args = ap.parse_args()

    out_dir = "results/ragas_eval"
    os.makedirs(out_dir, exist_ok=True)

    # 1) load data
    passages_df, test_df = robust_load_miniwiki()
    passages_df = inspect_and_deduplicate(passages_df)
    passages = passages_df["text"].tolist()
    qs = test_df["question"].astype(str).tolist()[: args.limit]
    golds = test_df["answer"].astype(str).tolist()[: args.limit]

    # 2) judge models for RAGAS
    judge_llm = build_local_llm()
    judge_emb = build_embeddings(args.embed_model)

    rows_compare = []

    # ---- NAIVE ----
    if args.only in ("both", "naive"):
        print("[RAGAS] Running NAIVE pipeline ...")
        preds_n, ctxs_n = run_naive(
            passages=passages,
            questions=qs,
            embed_model_name=args.embed_model,
            top_k=args.topk,
            prompt_style=args.prompt,
        )
        df_naive = ragas_score(qs, preds_n, ctxs_n, golds, judge_llm, judge_emb)
        df_naive.to_csv(os.path.join(out_dir, "naive_items.csv"), index=False)
        mean_naive = df_naive.mean(numeric_only=True).round(4).to_dict()
        with open(os.path.join(out_dir, "summary_naive.json"), "w", encoding="utf-8") as f:
            json.dump(mean_naive, f, ensure_ascii=False, indent=2)
        print("[RAGAS] NAIVE means:", mean_naive)
        rows_compare.append({"system": "naive", **mean_naive})

    # ---- ENHANCED ----
    if args.only in ("both", "enhanced"):
        if not HAVE_ENHANCED:
            raise RuntimeError("Enhanced pipeline requested but EnhancedRAG not found.")
        print("[RAGAS] Running ENHANCED pipeline (HyDE & reranker) ...")
        preds_e, ctxs_e = run_enhanced_wrapped(
            passages=passages,
            questions=qs,
            top_k=args.topk,
            prompt_style=args.prompt,
            embed_model_name=args.embed_model,
            strategy=args.strategy,
            pool=args.pool,
            use_hyde=args.use_hyde,
            use_reranker=args.use_reranker,
            rerank_pool=args.rerank_pool,
        )
        df_enh = ragas_score(qs, preds_e, ctxs_e, golds, judge_llm, judge_emb)
        df_enh.to_csv(os.path.join(out_dir, "enhanced_items.csv"), index=False)
        mean_enh = df_enh.mean(numeric_only=True).round(4).to_dict()
        with open(os.path.join(out_dir, "summary_enhanced.json"), "w", encoding="utf-8") as f:
            json.dump(mean_enh, f, ensure_ascii=False, indent=2)
        print("[RAGAS] ENHANCED means:", mean_enh)
        rows_compare.append({"system": "enhanced", **mean_enh})

    # 3) side-by-side summary
    if rows_compare:
        side = pd.DataFrame(rows_compare)
        side.to_csv(os.path.join(out_dir, "summary_compare.csv"), index=False)
        print("\nSaved:", os.path.join(out_dir, "summary_compare.csv"))

if __name__ == "__main__":
    # For deterministic CPU usage on some BLAS stacks
    os.environ.setdefault("RAGAS_MAX_WORKERS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("RAGAS_MAX_RETRIES", "2")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
