# src/evaluation.py
from __future__ import annotations
import json, os, re, string
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from .utils import robust_load_miniwiki, build_faiss_index, search_top_k, inspect_and_deduplicate

GEN_MODEL = "google/flan-t5-base"

PROMPTS = {
    "instruction": """Answer the question using only the context. If unknown, say "I don't know".
Question: {q}
Context:
{ctx}
Answer:""",
    "persona_scientist": """You are a precise research assistant. Cite facts only from the context. If missing, reply "I don't know".
Q: {q}
Context:
{ctx}
A:""",
    "cot": """Answer step-by-step using only the context. If insufficient, say "I don't know".
Question: {q}
Context:
{ctx}
Let's reason step by step and conclude with the final answer after `Final:`.
"""
}

def load_generator(model_name: str = GEN_MODEL):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # If you have GPU/MPS, you can pass device_map="auto". To silence the warning, set device explicitly.
    return pipeline("text2text-generation", model=mdl, tokenizer=tok, device_map="auto")

# ---------------------------
# Local SQuAD-style metrics
# ---------------------------

_ARTICLES = {"a", "an", "the"}
_PUNCT = set(string.punctuation)

def _normalize(text: str) -> str:
    """Lowercase, remove punctuation, remove articles, and normalize whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    # remove punctuation
    text = "".join(ch for ch in text if ch not in _PUNCT)
    # remove articles
    tokens = text.split()
    tokens = [t for t in tokens if t not in _ARTICLES]
    # normalize whitespace
    return " ".join(tokens).strip()

def _f1(pred: str, gold: str) -> float:
    """Token-level F1 between prediction and gold (SQuAD-style)."""
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    tmp = {}
    for t in gold_tokens:
        tmp[t] = tmp.get(t, 0) + 1
    for t, c in tmp.items():
        if t in common:
            overlap += min(common[t], c)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def _exact_match(pred: str, gold: str) -> float:
    """Exact match after normalization."""
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0

def local_squad_metrics(preds: List[str], golds: List[str]) -> Dict[str, float]:
    em = sum(_exact_match(p, g) for p, g in zip(preds, golds)) / len(golds)
    f1 = sum(_f1(p, g) for p, g in zip(preds, golds)) / len(golds)
    # Convert to percentages to match common reporting, or keep as 0~1. Here we keep percentages.
    return {"exact_match": round(em * 100, 2), "f1": round(f1 * 100, 2)}

# ---------------------------

def postprocess_prediction(text: str) -> str:
    # If CoT is used, extract after "Final:"
    if "Final:" in text:
        return text.split("Final:")[-1].strip()
    return text.strip()

def eval_run(prompt_style: str = "instruction", top_k: int = 1, limit: int = 100) -> Dict[str, float]:
    passages_df, test_df = robust_load_miniwiki()
    passages_df = inspect_and_deduplicate(passages_df)
    passages = passages_df["text"].tolist()
    idx, embed_model, _ = build_faiss_index(passages)
    gen = load_generator()

    qs = test_df["question"].astype(str).tolist()[:limit]
    golds = test_df["answer"].astype(str).tolist()[:limit]

    preds = []
    for q in tqdm(qs, desc=f"Evaluating {prompt_style}, top_k={top_k}"):
        hits = search_top_k(q, idx, embed_model, passages, k=top_k)
        ctx = "\n\n---\n\n".join([h[2] for h in hits]) if hits else ""
        prompt = PROMPTS[prompt_style].format(q=q, ctx=ctx)
        out = gen(prompt, max_new_tokens=128)[0]["generated_text"]
        preds.append(postprocess_prediction(out))

    scores = local_squad_metrics(preds, golds)
    os.makedirs("results", exist_ok=True)
    outpath = "results/naive_results.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump({"prompt_style": prompt_style, "top_k": top_k, "scores": scores}, f, ensure_ascii=False, indent=2)
    return scores

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    combined = {"top_k": 1, "limit": 100, "scores_by_prompt": {}}

    for style in ["instruction", "persona_scientist"]:
        _ = eval_run(prompt_style=style, top_k=1, limit=100)
        with open("results/naive_results.json", "r", encoding="utf-8") as f:
            obj = json.load(f)
        combined["scores_by_prompt"][style] = obj["scores"]

    with open("results/naive_results.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print("Wrote combined results to results/naive_results.json")