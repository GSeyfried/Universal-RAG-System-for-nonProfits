#!/usr/bin/env python3
"""
2_text_splitter.py

Reads a JSON list of {"metadata":…, "text":…} chunks, splits each `text` into
sub-chunks of roughly `chunk_size` characters with `chunk_overlap`, and writes
out a new JSON array of {"metadata":…, "text":…}.

python .\scripts\2_text_splitter.py --input_file chunks.json --output_file split_chunks.json --chunk_size 512 --chunk_overlap 64
"""
import argparse
import json
import tiktoken
from langchain.text_splitter import TokenTextSplitter

def merge_short(pieces, tokenizer, min_tokens=50):
    merged = []
    for pc in pieces:
        # count tokens via tiktoken
        length = len(tokenizer.encode(pc))
        if merged and length < min_tokens:
            merged[-1] = merged[-1] + "\n" + pc
        else:
            merged.append(pc)
    return merged

def split_chunks(input_path, output_path, chunk_size, chunk_overlap):
    # load pre-chunked JSON
    with open(input_path, "r", encoding="utf-8") as f:
        originals = json.load(f)

    # prepare tokenizer and splitter
    tokenizer = tiktoken.get_encoding("cl100k_base")
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    out_chunks = []
    new_id = 0

    for entry in originals:
        meta = entry["metadata"]
        text = entry["text"]

        # prepend section context
        hdr = " | ".join(meta.get("section_hierarchy", []))
        if len(hdr) > 100:
            hdr = hdr[:100] + "…"
        full = f"[{hdr}]\n{text}" if hdr else text

        # split into pieces
        pieces = splitter.split_text(full)
        # merge any pieces under min_tokens back into previous
        pieces = merge_short(pieces, tokenizer, min_tokens=50)

        for pc in pieces:
            out_chunks.append({
                "metadata": { **meta, "chunk_id": new_id },
                "text": pc
            })
            new_id += 1

    # write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_chunks, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(out_chunks)} split chunks -> {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_file",    default="chunks.json",
                   help="Input JSON from 1_pdf_processor")
    p.add_argument("--output_file",   default="split_chunks.json",
                   help="Where to write split chunks")
    p.add_argument("--chunk_size",    type=int, default=512,
                   help="Max tokens per chunk")
    p.add_argument("--chunk_overlap", type=int, default=64,
                   help="Tokens to overlap between chunks")
    args = p.parse_args()

    split_chunks(
        args.input_file,
        args.output_file,
        args.chunk_size,
        args.chunk_overlap
    )