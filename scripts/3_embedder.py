#!/usr/bin/env python3
"""
3_embedder.py

Generate embeddings for split text chunks.

Usage (bash):
    python .\scripts\3_embedder.py --input_file split_chunks.json --output_file embeddings.json --batch_size 16 --delay 1.0
"""
import os, json, time, argparse
from openai import OpenAI
from load_config import initialize_config, get_env_variable

cfg = initialize_config()
api_key = get_env_variable("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
model = cfg["openai"]["embedding_model"]

def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_embeddings(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file",  default="split_chunks.json")
    p.add_argument("--output_file", default="embeddings.json")
    p.add_argument("--model",       default="text-embedding-ada-002")
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--delay",       type=float, default=1.0,
                   help="Seconds to sleep between batches")
    args = p.parse_args()

    # ── Load config & initialize client ──────────────────────────────────────
    cfg     = initialize_config()
    api_key = get_env_variable("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env or secrets")
    print(f"API key loaded: {'yes' if api_key else 'no'}")
    print(f"Embedding model: {args.model}")
    client = OpenAI(api_key=api_key)

    # ── Load chunks ─────────────────────────────────────────────────────────
    chunks = load_chunks(args.input_file)
    total  = len(chunks)
    print(f"Loaded {total} chunks from {args.input_file}")

    # ── Resume existing embeddings if any ───────────────────────────────────
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            embeddings_data = json.load(f)
        start_idx = len(embeddings_data)
        print(f"Resuming from index {start_idx}")
    else:
        embeddings_data = []
        start_idx = 0
        print("Starting fresh embedding run")

    # ── Batch loop ──────────────────────────────────────────────────────────
    for i in range(start_idx, total, args.batch_size):
        batch = chunks[i : i + args.batch_size]
        texts = [c["text"] for c in batch]
        print(f"Embedding batch {i}–{i+len(batch)-1} ({len(batch)} chunks)", flush=True)

        try:
            resp = client.embeddings.create(
                input=texts,
                model=args.model
            )
            for entry, result in zip(batch, resp.data):
                embeddings_data.append({
                    "metadata": entry["metadata"],
                    "text":      entry["text"],
                    "embedding": result.embedding
                })
            save_embeddings(args.output_file, embeddings_data)
            print(f"   [OK] Saved up to {len(embeddings_data)} embeddings", flush=True)
        except Exception as e:
            print(f"   [FAILED] Error on batch {i}: {e}", flush=True)

        time.sleep(args.delay)

    print("Embedding generation complete!")

if __name__ == "__main__":
    main()
