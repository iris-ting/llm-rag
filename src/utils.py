from __future__ import annotations
import os, json, re
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

HF_DATASET_ID = "rag-datasets/rag-mini-wikipedia"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def robust_load_miniwiki() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the 'rag-mini-wikipedia' dataset by explicitly selecting configs:
      - 'text-corpus'  -> passages (train)
      - 'question-answer' -> QA set (prefer 'test', else 'validation')
    Falls back to local parquet files if needed.
    """
    # --- Preferred path: load from Hugging Face with explicit configs ---
    try:
        passages_ds = load_dataset(HF_DATASET_ID, "text-corpus")
        qa_ds = load_dataset(HF_DATASET_ID, "question-answer")

        # pick splits robustly
        # text-corpus usually has 'train'
        passages_split = "train" if "train" in passages_ds else list(passages_ds.keys())[0]
        # question-answer may have 'test' or 'validation'
        if "test" in qa_ds:
            qa_split = "test"
        elif "validation" in qa_ds:
            qa_split = "validation"
        else:
            qa_split = list(qa_ds.keys())[0]

        passages_df = passages_ds[passages_split].to_pandas()
        test_df = qa_ds[qa_split].to_pandas()

    except Exception:
        # --- Fallback: try local parquet files in ./data/ ---
        # Expect: data/passages.parquet and data/test.parquet
        p_path = os.path.join("data", "passages.parquet")
        t_path = os.path.join("data", "test.parquet")
        if not (os.path.exists(p_path) and os.path.exists(t_path)):
            raise RuntimeError(
                "Failed to load from Hugging Face and local parquet files not found at data/*. "
                "Ensure internet access to download the dataset or place 'passages.parquet' and 'test.parquet' under ./data."
            )
        passages_df = pd.read_parquet(p_path)
        test_df = pd.read_parquet(t_path)

    # --- Normalize columns ---
    passages_df.columns = [c.lower() for c in passages_df.columns]
    test_df.columns = [c.lower() for c in test_df.columns]

    # Detect passage text column in text-corpus (usually 'text', sometimes 'content'/'context')
    passage_text_col = None
    for c in ["text", "passage", "content", "context", "paragraph", "wiki", "body"]:
        if c in passages_df.columns:
            passage_text_col = c
            break
    if passage_text_col is None:
        str_cols = [c for c in passages_df.columns if passages_df[c].dtype == object]
        passages_df["text"] = passages_df[str_cols].astype(str).agg(" ".join, axis=1)
        passage_text_col = "text"

    # Detect question / answer columns in QA split
    q_col = None
    for c in ["question", "query", "q"]:
        if c in test_df.columns:
            q_col = c
            break
    a_col = None
    # in some HF exports answers may appear as a plain string or list/dict
    for c in ["answer", "answers", "gold", "label"]:
        if c in test_df.columns:
            a_col = c
            break

    # Minimal coercion if formats vary
    if q_col is None:
        str_cols = [c for c in test_df.columns if test_df[c].dtype == object]
        q_col = str_cols[0] if str_cols else test_df.columns[0]

    if a_col is None:
        # create empty answers if truly missing so the pipeline can run
        test_df["answer"] = ""
        a_col = "answer"

    # If answers look like lists/dicts, convert to a single string
    def _to_text(x):
        if isinstance(x, str):
            return x
        if isinstance(x, dict) and "text" in x:
            # e.g., {"text": ["abc"], "answer_start":[0]}
            txt = x.get("text")
            return txt[0] if isinstance(txt, list) and txt else str(x)
        if isinstance(x, list) and x:
            return x[0] if isinstance(x[0], str) else str(x[0])
        return str(x)

    test_df[a_col] = test_df[a_col].apply(_to_text)

    passages_df = passages_df.rename(columns={passage_text_col: "text"})
    test_df = test_df.rename(columns={q_col: "question", a_col: "answer"})

    # Return only necessary columns
    return passages_df[["text"]].dropna(), test_df[["question", "answer"]].dropna()

def build_faiss_index(passages: List[str], model_name: str = EMBED_MODEL_NAME):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(passages, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity since normalized)
    index.add(embeddings)
    return index, model, embeddings

def search_top_k(
    query: str,
    index,
    embed_model: SentenceTransformer,
    passages: List[str],
    k: int = 5
) -> List[Tuple[int, float, str]]:
    q_vec = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_vec, k)
    hits = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        if idx == -1: continue
        hits.append((int(idx), float(score), passages[idx]))
    return hits

def inspect_and_deduplicate(passages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Print dataset size, detect duplicate passages, drop them, and print size again.
    """
    print(f"Original passages count: {len(passages_df)}")
    dup_count = passages_df.duplicated(subset=["text"]).sum()
    print(f"Number of duplicated passages: {dup_count}")
    passages_df = passages_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Passages count after removing duplicates: {len(passages_df)}")
    return passages_df