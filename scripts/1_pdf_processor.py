#!/usr/bin/env python3
"""
1_pdf_processor.py

Extracts text from PDFs and chunks it into sections using a data-driven
font-size classification (header/subheader/body/remove).

Usage:
  python scripts/1_pdf_processor.py --pdf_folder ./pdfs --classify_json font_inspector_output.json --output_file chunks.json
"""
import os, re, json, argparse
import fitz  # PyMuPDF
from math import inf

# Boilerplate page‐stamp regex
PAGE_STAMP = re.compile(r"\d{2}/\d{2}/\d{4}\s*Page\s*\d+")
DOT_LEADER = re.compile(r"\.{2,}")

def load_classification(path):
    data = json.load(open(path, encoding="utf-8"))
    size_map = {entry["font_size"]: entry["classification"]
                for entry in data["font_sizes"]}
    remove_texts = set(data["remove_texts"])
    return size_map, remove_texts

def classify_size(size, size_map):
    # Try exact match
    sz = round(size, 2)
    if sz in size_map:
        return size_map[sz]
    # Otherwise nearest
    best, bd = None, inf
    for s in size_map:
        d = abs(s - size)
        if d < bd:
            best, bd = s, d
    return size_map[best]

def extract_chunks_from_pdf(pdf_path, size_map, remove_texts):
    doc = fitz.open(pdf_path)
    chunks = []
    current_h1 = current_h2 = None
    chunk_id = 0
    pdf_name = os.path.basename(pdf_path)

    def flush_buffer(page_idx):
        nonlocal buffer, chunk_id
        if buffer:
            chunks.append({
                "metadata": {
                    "chunk_id":        chunk_id,
                    "pdf_name":        pdf_name,
                    "page_number":     page_idx+1,
                    "section_hierarchy":[h for h in (current_h1, current_h2) if h],
                    "headers":         {"h1": current_h1, "h2": current_h2},
                    "is_appendix":     False
                },
                "text": "\n".join(buffer).strip()
            })
            chunk_id += 1
            buffer.clear()

    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        blocks = page.get_text("dict")["blocks"]
        buffer = []

        for block in blocks:
            if block["type"] != 0: continue

            # assemble lines + track max font
            lines, max_font = [], 0
            for line in block["lines"]:
                txts = [s["text"].strip() for s in line["spans"] if s["text"].strip()]
                for s in line["spans"]:
                    max_font = max(max_font, s["size"])
                if txts:
                    lines.append(" ".join(txts))
            if not lines:
                continue

            full_text = " ".join(lines)
            # skip boilerplate
            if (full_text in remove_texts
                or PAGE_STAMP.match(full_text)
                or DOT_LEADER.search(full_text)
                or len(full_text.split()) < 3):
                continue

            cls = classify_size(max_font, size_map)
            if cls == "remove":
                continue
            if cls == "header":
                flush_buffer(page_idx)
                current_h1, current_h2 = full_text, None
                continue
            if cls == "subheader":
                flush_buffer(page_idx)
                current_h2 = full_text
                continue

            # body
            buffer.append(full_text)

        flush_buffer(page_idx)

    doc.close()
    return chunks

def process_pdfs(pdf_folder, classify_json, output_file):
    size_map, remove_texts = load_classification(classify_json)
    all_chunks = []
    for fname in sorted(os.listdir(pdf_folder)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_folder, fname)
        print(f"Processing {path}…")
        all_chunks.extend(
            extract_chunks_from_pdf(path, size_map, remove_texts)
        )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_chunks)} chunks -> {output_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdf_folder",    required=True,
                   help="Directory containing PDFs")
    p.add_argument("--classify_json", required=True,
                   help="font_inspector JSON with classifications")
    p.add_argument("--output_file",   default="chunks.json",
                   help="Where to write the chunks")
    args = p.parse_args()

    process_pdfs(args.pdf_folder, args.classify_json, args.output_file)
