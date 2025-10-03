# Enhancement Options: both reranking and HyDE (Hypothetical Document Embeddings) are default enabled.

# src/enhanced_rag.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import os, time
import numpy as np

from sentence_transformers import CrossEncoder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from .utils import robust_load_miniwiki, build_faiss_index, search_top_k, inspect_and_deduplicate
from .evaluation import PROMPTS, postprocess_prediction  # reuse your prompts/postprocess

# ---------------- helpers ----------------

def trim_to_budget(text: str, tokenizer, max_tokens: int = 512) -> str:
    """Trim a string to a token budget using the tokenizer."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

def hyde_document(question: str, gen_pipeline) -> str:
    """Generate a short pseudo passage for HyDE retrieval."""
    prompt = (
        "Write a concise pseudo passage (2-3 sentences) that could answer the question.\n"
        "Do not include citations.\n\n"
        f"Question: {question}\n\nPassage:"
    )
    out = gen_pipeline(prompt, max_new_tokens=120, truncation=True)[0]["generated_text"]
    return out.strip()

def mmr_select(
    q_vec: np.ndarray,
    cand_ids: List[int],
    emb_matrix: np.ndarray,
    k: int,
    lambda_mult: float = 0.8,
) -> List[int]:
    """MMR over candidate indices. Vectors must be L2-normalized."""
    if len(cand_ids) <= k:
        return list(cand_ids)
    selected: List[int] = []
    remaining = set(cand_ids)

    # relevance to query
    rel = {i: float(np.dot(emb_matrix[i], q_vec)) for i in cand_ids}

    # pick the most relevant first
    first = max(remaining, key=lambda i: rel[i])
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        def score(i: int) -> float:
            redundancy = max(float(np.dot(emb_matrix[i], emb_matrix[j])) for j in selected)
            return lambda_mult * rel[i] - (1 - lambda_mult) * redundancy
        best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)

    return selected

# ---------------- class ----------------

class EnhancedRAG:
    """
    Minimal class wrapper for the enhanced RAG pipeline.
    Methods used by RAGAS script:
      - load_corpus()
      - build_index()
      - rewrite_query(q): returns rewritten (HyDE) query or original
      - retrieve(q, top_k): returns a List[str] of passages
      - generate(question, passages): returns a string answer
    """

    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        gen_model: str = "google/flan-t5-base",
        top_k_default: int = 5,
        strategy: str = "concat",           # "concat" or "mmr"
        pool: int = 50,                      # candidate pool for retrieval
        use_hyde: bool = True,
        use_reranker: bool = True,
        rerank_pool: int = 50,
        prompt_style: str = "instruction",
    ):
        self.embed_model_name = embed_model
        self.gen_model_name = gen_model
        self.top_k_default = top_k_default
        self.strategy = strategy
        self.pool = pool
        self.use_hyde = use_hyde
        self.use_reranker = use_reranker
        self.rerank_pool = rerank_pool
        self.prompt_style = prompt_style

        # runtime objects
        self.passages: List[str] = []
        self.index = None
        self.embed_model = None
        self.emb_matrix: Optional[np.ndarray] = None
        self.gen = None
        self.reranker: Optional[CrossEncoder] = None

    # ---------- setup ----------

    def load_corpus(self):
        """Load and deduplicate passages into memory."""
        passages_df, _ = robust_load_miniwiki()
        passages_df = inspect_and_deduplicate(passages_df)
        self.passages = passages_df["text"].tolist()

    def build_index(self):
        """Build FAISS index and load generator/reranker."""
        self.index, self.embed_model, self.emb_matrix = build_faiss_index(
            self.passages, model_name=self.embed_model_name
        )
        tok = AutoTokenizer.from_pretrained(self.gen_model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(self.gen_model_name)
        self.gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, device_map="auto")
        if self.use_reranker:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    # ---------- operations ----------

    def rewrite_query(self, q: str) -> str:
        """Return HyDE rewritten query if enabled; otherwise return original."""
        if self.use_hyde and self.gen is not None:
            return hyde_document(q, self.gen)
        return q

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve passages given a (possibly rewritten) query.
        Returns a list of passage strings.
        """
        assert self.index is not None and self.embed_model is not None, "Index not built. Call build_index()."
        k = top_k or self.top_k_default
        prefetch = max(self.pool, k)

        # vector retrieval for a pool
        hits = search_top_k(query, self.index, self.embed_model, self.passages, k=prefetch)
        cand_ids = [h[0] for h in hits]

        # rerank (optional)
        if self.reranker is not None and len(cand_ids) > 0:
            pairs = [(query, self.passages[i]) for i in cand_ids[: self.rerank_pool]]
            scores = self.reranker.predict(pairs)
            order = np.argsort(-np.array(scores))
            cand_ids = [cand_ids[i] for i in order]

        # final select: concat or MMR
        if self.strategy == "mmr":
            q_vec = self.embed_model.encode([query], normalize_embeddings=True)[0]
            picked = mmr_select(q_vec, cand_ids[: max(20, k)], self.emb_matrix, k=k, lambda_mult=0.8)
        else:
            picked = cand_ids[:k]

        return [self.passages[i] for i in picked]

    def generate(self, question: str, passages: List[str]) -> str:
        """Generate an answer using the provided passages."""
        assert self.gen is not None, "Generator not loaded. Call build_index()."
        ctx = "\n\n---\n\n".join(passages)
        prompt = PROMPTS[self.prompt_style].format(q=question, ctx=ctx)
        prompt = trim_to_budget(prompt, self.gen.tokenizer, max_tokens=512)
        out = self.gen(prompt, max_new_tokens=128, truncation=True)[0]["generated_text"]
        return postprocess_prediction(out)

# ------------- optional CLI demo -------------

def main():
    """Tiny demo: build and answer one random question using enhanced pipeline."""
    import pandas as pd
    from random import randint

    rag = EnhancedRAG(top_k_default=5, strategy="mmr", use_hyde=True, use_reranker=True)
    rag.load_corpus()
    rag.build_index()

    # load a few questions just for demo
    _, test_df = robust_load_miniwiki()
    row = test_df.sample(1).iloc[0]
    q = str(row["question"]); gold = str(row["answer"])

    rewritten = rag.rewrite_query(q)
    ctxs = rag.retrieve(rewritten, top_k=5)
    pred = rag.generate(q, ctxs)

    print("\nQuestion:", q)
    print("Prediction:", pred)
    print("Gold:", gold)

if __name__ == "__main__":
    main()
