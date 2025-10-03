from __future__ import annotations
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from .utils import robust_load_miniwiki, build_faiss_index, search_top_k, inspect_and_deduplicate

GEN_MODEL = "google/flan-t5-base"

PROMPT_TMPL = """You are a helpful assistant. Use ONLY the provided context to answer.
If the answer is not in the context, say "I don't know".

Question: {question}

Context:
{context}

Answer:"""

def load_generator(model_name: str = GEN_MODEL):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, device_map="auto")
    return gen

def answer_with_top1(gen, question: str, context: str, max_new_tokens: int = 128) -> str:
    prompt = PROMPT_TMPL.format(question=question, context=context)
    out = gen(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"].strip()
    return out

def main_build_and_demo():
    # 1) Load dataset
    passages_df, test_df = robust_load_miniwiki()
    passages_df = inspect_and_deduplicate(passages_df) # check duplicated passages
    passages = passages_df["text"].tolist()

    # 2) Build FAISS index
    faiss_index, embed_model, _ = build_faiss_index(passages)

    # 3) Load generator
    gen = load_generator()

    # 4) Demo on a random sample
    sample = test_df.sample(1).iloc[0]
    q = str(sample["question"])
    gold = str(sample["answer"])

    hits = search_top_k(q, faiss_index, embed_model, passages, k=5)
    top_context = hits[0][2] if hits else ""
    pred = answer_with_top1(gen, q, top_context)

    print("\nQuestion:", q)
    print("Top-1 Context (truncated):", top_context[:300], "...")
    print("Prediction:", pred)
    print("Gold:", gold)

if __name__ == "__main__":
    main_build_and_demo()
