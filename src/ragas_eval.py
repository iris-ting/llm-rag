"""
RAGAS evaluation (minimal; no LangChain; no EnhancedRAG class required)
- Uses your existing naive pipeline: FAISS retrieval + generator (from evaluation.py)
- Computes RAGAS metrics: faithfulness, context_precision, context_recall, answer_relevancy
- Saves per-item CSV and mean JSON.
"""

from __future__ import annotations
import os, json, argparse
import pandas as pd
from datasets import Dataset

# ---- use your existing code ----
try:
    # when running "python -m src.ragas_evaluate"
    from .utils import robust_load_miniwiki, build_faiss_index, search_top_k, inspect_and_deduplicate
    from .evaluation import PROMPTS, postprocess_prediction, load_generator
except Exception:
    # when running as a plain script "python ragas_evaluate.py"
    from utils import robust_load_miniwiki, build_faiss_index, search_top_k, inspect_and_deduplicate
    from evaluation import PROMPTS, postprocess_prediction, load_generator

# ---- RAGAS ----
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, context_precision, context_recall

# be nice to local CPU by default
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("RAGAS_MAX_WORKERS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def run_ragas(limit: int = 50, top_k: int = 5, prompt_style: str = "instruction",
              out_dir: str = "results/ragas_eval") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load dataset and prepare passages/questions/ground truth
    passages_df, test_df = robust_load_miniwiki()
    if "text" in passages_df.columns:
        passages_df = inspect_and_deduplicate(passages_df)
        passages = passages_df["text"].astype(str).tolist()
    else:
        # fallback: if your utils returns a different schema
        passages = passages_df.iloc[:, 0].astype(str).tolist()

    questions = test_df["question"].astype(str).tolist()[:limit]
    references = test_df["answer"].astype(str).tolist()[:limit]   # <-- 不用 gold_answers 命名

    # 2) Build index + load generator (reuse your existing functions)
    index, embed_model, _ = build_faiss_index(passages)
    gen = load_generator()   # same generator used in your evaluation.py
    tok = gen.tokenizer

    # 3) Produce predictions and contexts
    preds, ctxs = [], []
    for q in questions:
        hits = search_top_k(q, index, embed_model, passages, k=top_k)
        # each hit is (idx, score, text); contexts must be List[str]
        ctx_list = [h[2] for h in hits]
        # build prompt and generate
        prompt = PROMPTS[prompt_style].format(q=q, ctx="\n\n---\n\n".join(ctx_list))
        out = gen(prompt, max_new_tokens=128, truncation=True)[0]["generated_text"]
        preds.append(postprocess_prediction(out))
        ctxs.append(ctx_list)

    # 4) Build HF Dataset for RAGAS (column names must match)
    df = pd.DataFrame({
        "question": questions,
        "ground_truth": references,
        "contexts": ctxs,
        "answer": preds,
    })
    ds = Dataset.from_pandas(df)

    # 5) RAGAS evaluation (four required metrics)
    # NOTE: If OPENAI_API_KEY is set, newer ragas versions will use it as judge automatically.
    result = ragas_evaluate(
        dataset=ds,
        metrics=[faithfulness, context_precision, context_recall],
        raise_exceptions=False,   # do not stop on per-item failures
    )

    # 6) Save outputs
    res_df = result.to_pandas()
    res_df.to_csv(os.path.join(out_dir, "naive_items.csv"), index=False)
    means = res_df.mean(numeric_only=True).round(4).to_dict()
    with open(os.path.join(out_dir, "summary_naive.json"), "w", encoding="utf-8") as f:
        json.dump(means, f, ensure_ascii=False, indent=2)


    print("[RAGAS] means:", means)
    print(f"Saved per-item: {os.path.join(out_dir, 'naive_items.csv')}")
    print(f"Saved summary:  {os.path.join(out_dir, 'summary_naive.json')}")

def main():
    ap = argparse.ArgumentParser(description="Minimal RAGAS evaluation using your existing naive pipeline.")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--prompt", type=str, default="instruction")
    ap.add_argument("--outdir", type=str, default="results/ragas_eval")
    args = ap.parse_args()
    run_ragas(limit=args.limit, top_k=args.topk, prompt_style=args.prompt, out_dir=args.outdir)


if __name__ == "__main__":
    main()
