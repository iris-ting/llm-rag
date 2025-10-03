# src/para_experiments.py
from __future__ import annotations
import os, json, time, argparse
from typing import List, Dict
import numpy as np
import pandas as pd

from .utils import robust_load_miniwiki, build_faiss_index, search_top_k, inspect_and_deduplicate
from .evaluation import load_generator, local_squad_metrics, postprocess_prediction, PROMPTS

# Default models cover >=2 different embedding sizes (384, 512)
DEFAULT_EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",              # 384-d
    "sentence-transformers/distiluse-base-multilingual-cased-v2",  # 512-d
]

def mmr_select(
    q_vec: np.ndarray,
    cand_ids: List[int],
    emb_matrix: np.ndarray,
    k: int,
    lambda_mult: float = 0.8,
) -> List[int]:
    """
    Maximal Marginal Relevance selection over candidate passage indices.
    Assumes q_vec and emb_matrix rows are L2-normalized (cosine = dot).
    """
    if len(cand_ids) <= k:
        return list(cand_ids)
    selected: List[int] = []
    remaining = set(cand_ids)

    # Precompute similarities to query
    sim_q = {i: float(np.dot(emb_matrix[i], q_vec)) for i in cand_ids}

    # 1) pick the most relevant first
    first = max(remaining, key=lambda i: sim_q[i])
    selected.append(first)
    remaining.remove(first)

    # 2) pick until k
    while len(selected) < k and remaining:
        def score(i: int) -> float:
            # relevance to query
            rel = sim_q[i]
            # max redundancy with selected
            red = max(float(np.dot(emb_matrix[i], emb_matrix[j])) for j in selected)
            return lambda_mult * rel - (1 - lambda_mult) * red

        best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)

    return selected

def build_context_for(
    question: str,
    passages: List[str],
    index,
    embed_model,
    emb_matrix: np.ndarray,
    top_k: int,
    strategy: str = "concat",
    prefetch: int = 20,
) -> str:
    """
    Build context string according to strategy:
      - concat: take top_k passages and concatenate
      - mmr: take top 'prefetch' then select K with MMR and concatenate
    """
    # Encode query
    q_vec = embed_model.encode([question], normalize_embeddings=True)[0]

    if strategy == "concat":
        hits = search_top_k(question, index, embed_model, passages, k=top_k)
        ctxs = [h[2] for h in hits]
        return "\n\n---\n\n".join(ctxs)

    elif strategy == "mmr":
        # get a richer candidate pool first
        prefetch = max(prefetch, top_k)
        q_vec_2d = q_vec.reshape(1, -1)
        D, I = index.search(q_vec_2d, prefetch)
        cand_ids = [i for i in I[0] if i != -1]
        picked = mmr_select(q_vec, cand_ids, emb_matrix, k=top_k, lambda_mult=0.8)
        ctxs = [passages[i] for i in picked]
        return "\n\n---\n\n".join(ctxs)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def run_one_setting(
    passages: List[str],
    questions: List[str],
    golds: List[str],
    embed_model_name: str,
    top_k: int,
    prompt_style: str,
    strategy: str,
) -> Dict[str, float]:
    """
    Build FAISS for a specific embedding model, evaluate with given top_k, prompt, and strategy.
    """
    t0 = time.time()
    index, embed_model, emb_matrix = build_faiss_index(passages, model_name=embed_model_name)
    build_secs = time.time() - t0

    # get embedding dimension if available
    try:
        embed_dim = embed_model.get_sentence_embedding_dimension()
    except Exception:
        embed_dim = emb_matrix.shape[1]

    gen = load_generator()

    preds = []
    t1 = time.time()
    for q in questions:
        ctx = build_context_for(
            question=q,
            passages=passages,
            index=index,
            embed_model=embed_model,
            emb_matrix=emb_matrix,
            top_k=top_k,
            strategy=strategy,
        )
        prompt = PROMPTS[prompt_style].format(q=q, ctx=ctx)
        out = gen(prompt, max_new_tokens=128)[0]["generated_text"]
        preds.append(postprocess_prediction(out))
    gen_secs = time.time() - t1

    scores = local_squad_metrics(preds, golds)
    return {
        "exact_match": scores["exact_match"],
        "f1": scores["f1"],
        "build_time_s": round(build_secs, 3),
        "gen_time_s": round(gen_secs, 3),
        "avg_time_per_q_s": round(gen_secs / max(1, len(questions)), 3),
        "embed_dim": int(embed_dim),
    }

def main():
    parser = argparse.ArgumentParser(description="Minimal Step-4 experiments over embeddings, top-k, and selection strategies.")
    parser.add_argument("--limit", type=int, default=100, help="Number of QA pairs to evaluate.")
    parser.add_argument("--topk", type=int, nargs="+", default=[3, 5, 10], help="List of top-k values (>=3).")
    parser.add_argument("--strategies", type=str, nargs="+", default=["concat", "mmr"],
                        help="Passage selection strategies: concat, mmr")
    parser.add_argument("--prompts", type=str, nargs="+", default=["instruction"],
                        help="Prompt styles to test (default: instruction for minimal requirement).")
    parser.add_argument("--embed-models", type=str, nargs="+", default=DEFAULT_EMBED_MODELS,
                        help="Embedding model names; include at least 2 different embedding sizes.")
    parser.add_argument("--out-csv", type=str, default="results/comparison_analysis.csv", help="Output CSV path.")
    parser.add_argument("--out-dir", type=str, default="results/para_runs", help="Directory to store per-run JSON.")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load and clean dataset
    passages_df, test_df = robust_load_miniwiki()
    passages_df = inspect_and_deduplicate(passages_df)
    passages = passages_df["text"].tolist()
    qs = test_df["question"].astype(str).tolist()[: args.limit]
    golds = test_df["answer"].astype(str).tolist()[: args.limit]

    rows = []
    total_runs = len(args.embed_models) * len(args.topk) * len(args.strategies) * len(args.prompts)
    run_idx = 1

    for embed_model_name in args.embed_models:
        for top_k in args.topk:
            for strategy in args.strategies:
                for prompt_style in args.prompts:
                    print(f"\n[{run_idx}/{total_runs}] embed='{embed_model_name}'  (k={top_k}, strategy={strategy}, prompt='{prompt_style}', limit={args.limit})")
                    metrics = run_one_setting(
                        passages=passages,
                        questions=qs,
                        golds=golds,
                        embed_model_name=embed_model_name,
                        top_k=top_k,
                        prompt_style=prompt_style,
                        strategy=strategy,
                    )
                    print(f"  -> dim={metrics['embed_dim']}  EM={metrics['exact_match']}  F1={metrics['f1']}  "
                          f"build={metrics['build_time_s']}s  gen={metrics['gen_time_s']}s")

                    # Save per-run JSON
                    run_json = {
                        "embed_model": embed_model_name,
                        "embed_dim": metrics["embed_dim"],
                        "top_k": top_k,
                        "strategy": strategy,
                        "prompt_style": prompt_style,
                        "limit": args.limit,
                        "scores": {"exact_match": metrics["exact_match"], "f1": metrics["f1"]},
                        "timing": {
                            "build_time_s": metrics["build_time_s"],
                            "gen_time_s": metrics["gen_time_s"],
                            "avg_time_per_q_s": metrics["avg_time_per_q_s"],
                        },
                    }
                    base = embed_model_name.split("/")[-1]
                    json_name = f"run_{prompt_style}_k{top_k}_{strategy}_{base}.json"
                    with open(os.path.join(args.out_dir, json_name), "w", encoding="utf-8") as f:
                        json.dump(run_json, f, ensure_ascii=False, indent=2)

                    rows.append({
                        "embed_model": embed_model_name,
                        "embed_dim": metrics["embed_dim"],
                        "top_k": top_k,
                        "strategy": strategy,
                        "prompt_style": prompt_style,
                        "limit": args.limit,
                        "exact_match": metrics["exact_match"],
                        "f1": metrics["f1"],
                        "build_time_s": metrics["build_time_s"],
                        "gen_time_s": metrics["gen_time_s"],
                        "avg_time_per_q_s": metrics["avg_time_per_q_s"],
                        "passages_after_dedup": len(passages),
                    })
                    run_idx += 1

    df = pd.DataFrame(rows).sort_values(by=["embed_dim", "embed_model", "top_k", "strategy"])
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved summary CSV to: {args.out_csv}")
    print(f"Per-run JSON files saved under: {args.out_dir}")

if __name__ == "__main__":
    main()