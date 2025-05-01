#!/usr/bin/env python3
"""
font_inspector.py

Inspects PDFs to build a universal font-size → classification map and
boilerplate remove list, then emits JSON for downstream use.

Usage:
  python scripts/font_inspector.py \
    --pdf_folder ./pdfs \
    --examples 5 \
    --remove_thresh 50 \
    --output_file font_inspector_output.json
"""
import os, re, json, argparse
import fitz  # PyMuPDF
from collections import defaultdict

def inspect_fonts(pdf_folder, max_examples, remove_thresh):
    # stats per font size
    font_stats = defaultdict(lambda: {
        "spans": 0,
        "chars": 0,
        "upper_spans": 0,
        "top_spans": 0,
        "page_spans": 0,
        "punct_chars": 0,
        "font_names": set(),
        "examples": []
    })
    # count each exact span for boilerplate detection
    text_counts = defaultdict(int)

    # Step 1: Gather raw stats
    for fname in sorted(os.listdir(pdf_folder)):
        if not fname.lower().endswith(".pdf"):
            continue
        doc = fitz.open(os.path.join(pdf_folder, fname))
        for page_idx, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            top_blocks = set(range(3))  # first 3 blocks as “header area”
            for blk_idx, block in enumerate(blocks):
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        txt = span["text"].strip()
                        if not txt:
                            continue
                        sz = round(span["size"], 2)
                        stat = font_stats[sz]

                        # update counts
                        stat["spans"] += 1
                        stat["chars"] += len(txt)
                        if sum(c.isupper() for c in txt) / len(txt) > 0.8:
                            stat["upper_spans"] += 1
                        if blk_idx in top_blocks:
                            stat["top_spans"] += 1
                        stat["page_spans"] += 1
                        # punctuation count
                        stat["punct_chars"] += sum(1 for c in txt if c in ".,:;!?")

                        # record font name
                        stat["font_names"].add(span["font"])

                        # examples
                        if len(stat["examples"]) < max_examples:
                            stat["examples"].append(txt.replace("\n"," ")[:100])

                        # boilerplate count
                        text_counts[txt] += 1
        doc.close()

    # total spans across all sizes
    total_spans = sum(s["spans"] for s in font_stats.values())
    rare_thresh = max(10, 0.005 * total_spans)

    # Step 2: Classify each size
    size_info = []
    for sz, s in sorted(font_stats.items()):
        avg_len   = s["chars"] / s["spans"]
        pct_upper = s["upper_spans"] / s["spans"]
        pct_top   = s["top_spans"] / s["page_spans"]
        # punctuation ratio
        pct_punct = s["punct_chars"] / s["chars"] if s["chars"] else 0
        # header score
        score     = pct_upper*0.5 + pct_top*0.3 - (avg_len/500)*0.2

        # decision tree
        if s["spans"] < rare_thresh:
            cls = "remove"
        elif pct_punct > 0.10:
            cls = "body"
        elif score >= 0.30 and any("Bold" in fn for fn in s["font_names"]) and pct_upper >= 0.5:
            cls = "header"
        elif score >= 0.10 and any("Bold" in fn for fn in s["font_names"]):
            cls = "subheader"
        else:
            cls = "body"

        size_info.append({
            "font_size":      sz,
            "font_names":     sorted(s["font_names"]),
            "spans":          s["spans"],
            "avg_length":     round(avg_len,1),
            "pct_upper":      round(pct_upper,2),
            "pct_top":        round(pct_top,2),
            "pct_punct":      round(pct_punct,2),
            "score":          round(score,2),
            "classification": cls,
            "examples":       s["examples"]
        })

    # Step 3: Identify repeated boilerplate lines to remove
    remove_texts = []
    for txt, cnt in text_counts.items():
        if cnt < remove_thresh:
            continue
        t = txt.strip()
        if len(t) < 3 or re.fullmatch(r'\W+', t):
            continue
        remove_texts.append(t)

    return size_info, remove_texts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf_folder",   default="./pdfs",
                   help="Folder containing PDFs")
    p.add_argument("--examples",     type=int, default=5,
                   help="Max examples per font size")
    p.add_argument("--remove_thresh",type=int, default=50,
                   help="Min occurrences to flag boilerplate")
    p.add_argument("--output_file",  default="font_inspector_output.json",
                   help="JSON output path")
    args = p.parse_args()

    sizes, removes = inspect_fonts(
        args.pdf_folder,
        args.examples,
        args.remove_thresh
    )
    out = {"font_sizes": sizes, "remove_texts": removes}
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote {len(sizes)} font-size entries and "
          f"{len(removes)} boilerplate lines to {args.output_file}")

if __name__ == "__main__":
    main()
