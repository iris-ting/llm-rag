# src/ragas_compare.py
"""
Run RAGAS for BOTH naive and enhanced RAG pipelines (no LangChain).
- naive: top-k retrieval -> prompt -> generate
- enhanced: (optional HyDE) -> retrieve candidates -> (MMR or concat, optional reranker) -> prompt -> generate
Metrics: faithfulness, context_precision, context_recall, answer_relevancy

Outputs:
  results/ragas_eval/naive_items_<prompt>.csv
  results/ragas_eval/summary_naive_<prompt>.json
  results/ragas_eval/enhanced_items_<prompt>.csv
  results/ragas_eval/summary_enhanced_<prompt>.json
  results/ragas_eval/summary_compare.csv
  results/ragas_eval/summary_compare_<prompt>.csv
"""

from __future__ import annotations
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset

# ---- reuse your existing code (same as src.evaluation uses) ----
try:
    # when running as "python -m src.ragas_compare"
    from .utils import robust_load_miniwiki, build_faiss_index, search_top_k, inspect_and_deduplicate
    from .evaluation import PROMPTS, postprocess_prediction, load_generator
except Exception:
    # when running as a plain script "python ragas_compare.py"
    from utils import robust_load_miniwiki, build_faiss_index, search_top_k, inspect_and_deduplicate
    from evaluation import PROMPTS, postprocess_prediction, load_generator

# ---- RAGAS ----
from ragas import evaluate as ragas_evaluate
try:
    from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
    ANSWER_METRIC = answer_relevancy
except Exception:
    from ragas.metrics import faithfulness, context_precision, context_recall
    from ragas.metrics import answer_correctness as ANSWER_METRIC  # fallback for older versions

# optional cross-encoder reranker (used only if available and --reranker given)
_CE = None
try:
    from sentence_transformers import CrossEncoder  # type: ignore
    _CE = CrossEncoder
except Exception:
    _CE = None

# be nice to local CPU by default
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("RAGAS_MAX_WORKERS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------- helpers ----------------
def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n


def mmr_select(query_emb: np.ndarray, cand_embs: np.ndarray, k: int, lambd: float = 0.5) -> list[int]:
    """Simple MMR on candidate embeddings (cosine). Return selected indices."""
    q = _norm(query_emb.reshape(1, -1))[0]
    C = _norm(cand_embs)   # [N, D]
    sim_q = C @ q          # [N]
    selected, remaining = [], list(range(C.shape[0]))
    while len(selected) < min(k, C.shape[0]):
        if not selected:
            idx = int(np.argmax(sim_q[remaining]))
            chosen = remaining.pop(idx)
            selected.append(chosen)
        else:
            S = C[selected]                               # [m, D]
            sim_div = (C[remaining] @ S.T).max(axis=1)   # [len(remaining)]
            score = lambd * sim_q[remaining] - (1 - lambd) * sim_div
            chosen = remaining[int(np.argmax(score))]
            remaining.remove(chosen)
            selected.append(chosen)
    return selected


def get_emb(embed_model, texts: list[str]) -> np.ndarray:
    """Try common interfaces to obtain embeddings from your embed_model."""
    if hasattr(embed_model, "encode"):
        return np.asarray(embed_model.encode(texts), dtype=np.float32)
    vecs = [np.array(embed_model(t)).astype(np.float32) for t in texts]
    return np.vstack(vecs)


def enhanced_retrieve(
    q: str,
    index,
    embed_model,
    passages: list[str],
    top_k: int = 5,
    strategy: str = "mmr",
    use_hyde: bool = False,
    use_reranker: bool = False,
    candidate_k: int = 20,
    generator=None,
) -> list[str]:
    """
    Return contexts for 'enhanced' retrieval:
      - optional HyDE: rewrite the query into a hypothetical answer
      - retrieve candidate_k docs
      - rerank (cross-encoder) or select with MMR / concat
    """
    query_for_search = q
    if use_hyde and generator is not None:
        hyde_prompt = f"Write a short possible answer to the question: {q}"
        hyp = generator(hyde_prompt, max_new_tokens=48, truncation=True)[0]["generated_text"]
        query_for_search = f"{q}\n{hyp}"

    hits = search_top_k(query_for_search, index, embed_model, passages, k=candidate_k)
    cand_texts = [h[2] for h in hits]

    # optional reranker
    if use_reranker and _CE is not None:
        try:
            ce = _CE("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [[q, t] for t in cand_texts]
            scores = ce.predict(pairs)
            order = np.argsort(scores)[::-1][:top_k]
            return [cand_texts[i] for i in order]
        except Exception:
            pass  # fall back to MMR/concat below

    if strategy == "concat":
        return cand_texts[:top_k]

    # default: MMR
    q_emb = get_emb(embed_model, [q])[0]
    cand_embs = get_emb(embed_model, cand_texts)
    sel = mmr_select(q_emb, cand_embs, k=top_k, lambd=0.5)
    return [cand_texts[i] for i in sel]


def run_one_variant(mode: str, questions: list[str], references: list[str],
                    passages: list[str], index, embed_model, generator,
                    prompt_style: str, top_k: int, out_dir: str,
                    strategy: str, use_hyde: bool, use_reranker: bool) -> dict:
    """Run one variant (naive or enhanced), evaluate with RAGAS, save files, return means."""
    preds, ctxs = [], []

    for q in questions:
        if mode == "naive":
            hits = search_top_k(q, index, embed_model, passages, k=top_k)
            ctx_list = [h[2] for h in hits]
        else:  # enhanced
            ctx_list = enhanced_retrieve(
                q, index, embed_model, passages,
                top_k=top_k, strategy=strategy, use_hyde=use_hyde,
                use_reranker=use_reranker, candidate_k=max(2 * top_k, 20),
                generator=generator
            )

        prompt = PROMPTS[prompt_style].format(q=q, ctx="\n\n---\n\n".join(ctx_list))
        out = generator(prompt, max_new_tokens=128, truncation=True)[0]["generated_text"]
        preds.append(postprocess_prediction(out))
        ctxs.append(ctx_list)

    # Build RAGAS dataset
    df = pd.DataFrame({
        "question": questions,
        "ground_truth": references,
        "contexts": ctxs,
        "answer": preds,
    })
    ds = Dataset.from_pandas(df)

    result = ragas_evaluate(
        dataset=ds,
        metrics=[faithfulness, context_precision, context_recall, ANSWER_METRIC],
        raise_exceptions=False
    )

    res_df = result.to_pandas()
    means = res_df.mean(numeric_only=True).round(4).to_dict()

    # Save with prompt suffix
    suffix = "_" + re.sub(r"[^0-9A-Za-z_-]+", "-", prompt_style.strip().lower())
    os.makedirs(out_dir, exist_ok=True)
    items_fp = os.path.join(out_dir, f"{mode}_items{suffix}.csv")
    summ_fp = os.path.join(out_dir, f"summary_{mode}{suffix}.json")

    res_df.to_csv(items_fp, index=False)
    with open(summ_fp, "w", encoding="utf-8") as f:
        json.dump(means, f, ensure_ascii=False, indent=2)

    print(f"[{mode.upper()}] means:", means)
    print(f"Saved per-item: {items_fp}")
    print(f"Saved summary:  {summ_fp}")
    return means


def main():
    ap = argparse.ArgumentParser(description="RAGAS compare naive vs enhanced (no LangChain).")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--prompt", type=str, default="instruction")
    ap.add_argument("--mode", type=str, default="both", choices=["naive", "enhanced", "both"])
    ap.add_argument("--strategy", type=str, default="mmr", choices=["mmr", "concat"])
    ap.add_argument("--hyde", action="store_true", help="use HyDE rewrite in enhanced")
    ap.add_argument("--reranker", action="store_true", help="use cross-encoder reranker if available")
    ap.add_argument("--outdir", type=str, default="results/ragas_eval")
    args = ap.parse_args()

    # 1) Load data
    passages_df, test_df = robust_load_miniwiki()
    if "text" in passages_df.columns:
        passages_df = inspect_and_deduplicate(passages_df)
        passages = passages_df["text"].astype(str).tolist()
    else:
        passages = passages_df.iloc[:, 0].astype(str).tolist()

    questions = test_df["question"].astype(str).tolist()[: args.limit]
    references = test_df["answer"].astype(str).tolist()[: args.limit]

    # 2) Build index + generator
    index, embed_model, _ = build_faiss_index(passages)
    generator = load_generator()

    # 3) Run variants
    means = {}
    if args.mode in ("naive", "both"):
        means["naive"] = run_one_variant(
            "naive", questions, references, passages, index, embed_model, generator,
            args.prompt, args.topk, args.outdir, args.strategy, False, False
        )
    if args.mode in ("enhanced", "both"):
        means["enhanced"] = run_one_variant(
            "enhanced", questions, references, passages, index, embed_model, generator,
            args.prompt, args.topk, args.outdir, args.strategy, args.hyde, args.reranker
        )

    # 4) Summary comparison (fixed file name + with suffix)
    if "naive" in means or "enhanced" in means:
        rows = []
        for k, v in means.items():
            r = {"mode": k}
            r.update(v)
            rows.append(r)
        comp = pd.DataFrame(rows)

        suffix = "_" + re.sub(r"[^0-9A-Za-z_-]+", "-", args.prompt.strip().lower())
        comp_dir = args.outdir
        os.makedirs(comp_dir, exist_ok=True)

        comp_fp_with_suffix = os.path.join(comp_dir, f"summary_compare{suffix}.csv")
        comp_fp_canonical = os.path.join(comp_dir, "summary_compare.csv")

        comp.to_csv(comp_fp_with_suffix, index=False)  # keep suffixed version
        comp.to_csv(comp_fp_canonical, index=False)    # canonical fixed name

        print(f"Saved comparison (with suffix): {comp_fp_with_suffix}")
        print(f"Saved comparison (canonical):   {comp_fp_canonical}")


if __name__ == "__main__":
    main()