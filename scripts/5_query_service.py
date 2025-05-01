#!/usr/bin/env python3
"""
query_service.py

CLI for RAG: embed a query, retrieve top-k chunks via FAISS, then call GPT-4
to generate an answer with citations.

Usage (bash):
python .\scripts\5_query_service.py --query "How is rent reasonableness determined?"
"""

import argparse
import json
import numpy as np
import faiss

from load_config import initialize_config, get_env_variable
from openai import OpenAI

# ─── Initialize once ─────────────────────────────────────────────────────
cfg     = initialize_config()
api_key = get_env_variable("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in .env or secrets")
client   = OpenAI(api_key=api_key)

# models & retrieval settings from config.yaml
EM_MODEL = cfg["openai"]["embedding_model"]
LLM_MODEL= cfg["openai"]["llm_model"]
TOP_K    = cfg["rag"]["top_k"]

# load your FAISS index + metadata
INDEX = faiss.read_index(cfg["faiss"]["index_file"])
with open(cfg["faiss"]["metadata_file"], "r", encoding="utf-8") as f:
    METADATA = json.load(f)


def embed_query(text: str) -> np.ndarray:
    """Generate a single embedding vector for `text`."""
    resp = client.embeddings.create(input=[text], model=EM_MODEL)
    return np.array(resp.data[0].embedding, dtype="float32")


def retrieve_chunks(query_emb: np.ndarray, k: int):
    """Return the top-k metadata entries most similar to query_emb."""
    D, I = INDEX.search(query_emb.reshape(1, -1), k)
    return [METADATA[i] for i in I[0]]


def format_citations(chunks_meta):
    """Turn each chunk’s metadata into a human‐readable § citation."""
    cites = []
    for m in chunks_meta:
        sect = " > ".join(m.get("section_hierarchy", [])) or "Unknown Section"
        cites.append(f"§ {sect} (p. {m['page_number']})")
    return cites


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help="Your question")
    p.add_argument("--top_k", type=int, default=None,
                   help="How many chunks to retrieve")
    args = p.parse_args()

    k = args.top_k or TOP_K
    print(f"🔍 Embedding query: “{args.query}”")
    print(f"⚙️  Retrieving top {k} chunks…")

    # 1) Embed
    q_emb = embed_query(args.query)

    # 2) Retrieve
    hits = retrieve_chunks(q_emb, k)

    # 3) Build context + citations
    context = "\n\n".join(h.get("text", "") for h in hits)
    cites   = format_citations(hits)

    # 4) Chat completion
    system = """You are an AI assistant whose sole knowledge is the provided context. Only answer using information in that context—do not draw on outside knowledge. If the answer cannot be found in the context, say “I’m sorry, I don’t see content for that question in the documents." Always list only the provided citations. """
    user    = f"Context:\n{context}\n\nQuestion: {args.query}"
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user}
        ]
    )
    answer = resp.choices[0].message.content.strip()

    # 5) Print
    print("\n--- Answer ---\n")
    print(answer)
    print("\n--- Sources ---")
    for c in cites:
        print(c)


if __name__ == "__main__":
    main()
