#!/usr/bin/env python3
"""
4_indexer.py

Build a FAISS vector index from embeddings.json and save metadata.json.

Usage (bash):
    python .\scripts\4_indexer.py --input_file embeddings.json --index_file faiss_index.bin --meta_file metadata.json
"""

import argparse
import json
import numpy as np
import faiss

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file",  default="embeddings.json",
                   help="JSON list with {metadata, text, embedding}")
    p.add_argument("--index_file",  default="faiss_index.bin",
                   help="Where to write the FAISS index")
    p.add_argument("--meta_file",   default="metadata.json",
                   help="Where to write chunk metadata+text")
    args = p.parse_args()

    # Load the embeddings
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract embedding vectors and metadata
    embeddings = np.array([item["embedding"] for item in data], dtype="float32")
    meta      = [
        {**item["metadata"], "text": item["text"]}
        for item in data
    ]

    # Build FAISS index (L2)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"Indexed {index.ntotal} vectors (dim={dim})")

    # Save index and metadata
    faiss.write_index(index, args.index_file)
    with open(args.meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved index -> {args.index_file}")
    print(f"Saved metadata -> {args.meta_file}")

if __name__ == "__main__":
    main()
