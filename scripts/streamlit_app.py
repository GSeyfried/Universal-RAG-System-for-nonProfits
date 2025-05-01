#!/usr/bin/env python3
r"""
Streamlit front-end with:
 • Two-phase pipeline (font inspector → confirm → full pipeline)
 • PDF viewer via Expander
 • Delete PDF + artifacts
 • No experimental_rerun
"""
import os
import shutil
import subprocess
import json
import base64
import sys
PY = sys.executable 

import streamlit as st
import numpy as np
import faiss

from load_config import initialize_config, get_env_variable
from openai import OpenAI

import hashlib
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# ─── Config & Client ───────────────────────────────────────────────────────
cfg       = initialize_config()
API_KEY   = get_env_variable("OPENAI_API_KEY")
if not API_KEY:
    st.error("OPENAI_API_KEY not set")
    st.stop()
client    = OpenAI(api_key=API_KEY)

PDF_FOLDER= cfg["pdf_folder"]
TMP_ROOT  = cfg["tmp_folder"]
SCRIPTS   = cfg["scripts"]
COST = cfg["cost"]

if "prompt_tokens"     not in st.session_state: st.session_state.prompt_tokens = 0
if "completion_tokens" not in st.session_state: st.session_state.completion_tokens = 0

# ─── Simple username / password gate ───────────────────────────────────────
secrets = st.secrets.get("auth", {})
if secrets:
    USER = secrets.get("username")
    PASS = secrets.get("password")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        with st.form("login"):
            st.subheader("🔒 Login required")
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            ok = st.form_submit_button("Log in")

            if ok:
                if u == USER and p == PASS:
                    st.session_state.authenticated = True
                    st.success("✅ Login successful. Please continue.")
                else:
                    st.error("❌ Invalid credentials.")

            st.stop()
            
# ─── Helpers ───────────────────────────────────────────────────────────────
def paths_for(pdf_name):
    stem   = os.path.splitext(pdf_name)[0]
    outdir = os.path.join(TMP_ROOT, stem)
    os.makedirs(outdir, exist_ok=True)
    return {
        "outdir":         outdir,
        "font_inspector": os.path.join(outdir, "font_inspector_output.json"),
        "chunks":         os.path.join(outdir, "chunks.json"),
        "split_chunks":   os.path.join(outdir, "split_chunks.json"),
        "embeddings":     os.path.join(outdir, "embeddings.json"),
        "faiss_index":    os.path.join(outdir, "faiss_index.bin"),
        "metadata":       os.path.join(outdir, "metadata.json"),
    }

# ─── Sidebar: select/upload/delete PDF ─────────────────────────────────────
st.sidebar.header("1. PDF Management")

# Upload
uploaded = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded:
    dest = os.path.join(PDF_FOLDER, uploaded.name)
    with open(dest, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success(f"Uploaded {uploaded.name}")
    # no rerun: just refresh list below

# List
pdfs = sorted(f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf"))
selected = st.sidebar.selectbox("Select a PDF", [""] + pdfs)
if not selected:
    st.sidebar.info("Upload or select a PDF to proceed")
    st.stop()

# Delete
if st.sidebar.button("🗑 Delete PDF and data"):
    pdf_path = os.path.join(PDF_FOLDER, selected)
    tmp_dir  = os.path.join(TMP_ROOT, os.path.splitext(selected)[0])
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    st.sidebar.success(f"Deleted {selected} and its data. Please refresh.")
    st.stop()

paths     = paths_for(selected)
processed = os.path.exists(paths["faiss_index"])

# ─── Sidebar: Settings ─────────────────────────────────────────────────────
st.sidebar.header("2. Settings")
# model selectors from config
em_model  = st.sidebar.selectbox(
    "Embedding model (probably dont touch)",
    cfg["openai"]["embedding_model_options"],
    index=cfg["openai"]["embedding_model_options"].index(cfg["openai"]["embedding_model"])
)
llm_model = st.sidebar.selectbox(
    "LLM model",
    cfg["openai"]["llm_model_options"],
    index=cfg["openai"]["llm_model_options"].index(cfg["openai"]["llm_model"])
)
top_k     = st.sidebar.number_input(
    "Top-k retrievers (higher value = more context and more expensive)",
    min_value=1, max_value=20,
    value=cfg["rag"]["top_k"]
)

# ─── Live cost meter ──────────────────────────────────────────────────────
prompt_tok   = st.session_state.prompt_tokens
compl_tok    = st.session_state.completion_tokens

p_rate = COST.get(f"{llm_model}_prompt", 0)
c_rate = COST.get(f"{llm_model}_completion", 0)

p_cost = prompt_tok / 1000 * p_rate
c_cost = compl_tok  / 1000 * c_rate
total  = p_cost + c_cost

st.info(
    f"💰 Session usage — Prompt **{prompt_tok}** tok, "
    f"Completion **{compl_tok}** tok  •  "
    f"Est. cost **${total:.4f}**"
)
# ─── Sidebar: Pipeline ─────────────────────────────────────────────────────
st.sidebar.header("3. Pipeline")

def run_step(cmd: list[str], label: str):
    """Run a subprocess, surface stderr inside Streamlit, stop on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"❌ {label} failed:\n\n```\n{result.stderr}\n```")
        st.stop()
    st.sidebar.write(f"✅ {label} finished")

# Phase 1: Font Inspector
if not os.path.exists(paths["font_inspector"]):
    if st.sidebar.button("▶️ Run Font Inspector"):
        run_step([
            PY, SCRIPTS["font_inspector"],
            "--pdf_folder",  PDF_FOLDER,
            "--output_file", paths["font_inspector"],
            "--examples",    "10",
            "--remove_thresh","50"
        ], "Font Inspector")
        st.sidebar.success("Font Inspector complete. Please review mapping below.")
# Phase 2: Confirm font mapping
elif not os.path.exists(paths["faiss_index"]):
    st.sidebar.info("Edit & confirm font mapping before proceeding")
    raw_json = open(paths["font_inspector"],"r",encoding="utf-8").read()
    edited   = st.sidebar.text_area(
        "Font-inspector output (JSON)",
        raw_json, height=300
    )
    if st.sidebar.button("✅ Confirm Mapping & Continue"):
        try:
            json.loads(edited)       
            with open(paths["font_inspector"], "w", encoding="utf-8") as f:
                f.write(edited)
            st.sidebar.success("Mapping saved. — running full pipeline…")

            # pdf_processor
            run_step([
                PY, SCRIPTS["pdf_processor"],
                "--pdf_folder",    PDF_FOLDER,
                "--classify_json", paths["font_inspector"],
                "--output_file",   paths["chunks"]
            ], "PDF Processor")
            st.sidebar.success("PDF Processed into text & metadata")
            # text_splitter
            run_step([
                PY, SCRIPTS["text_splitter"],
                "--input_file",    paths["chunks"],
                "--output_file",   paths["split_chunks"],
                "--chunk_size",    "512",
                "--chunk_overlap", "64"
            ], "Text Splitter")
            st.sidebar.success("Text Split into chunks successfully")
            # embedder
            run_step([
                PY, SCRIPTS["embedder"],
                "--input_file",    paths["split_chunks"],
                "--output_file",   paths["embeddings"],
                "--model",         em_model,
                "--batch_size",    "16",
                "--delay",         "1.0"
            ], "Embedder")
            st.sidebar.success("Text has been convertd into vectors")
            # faiss_indexer
            run_step([
                PY, SCRIPTS["faiss_indexer"],
                "--input_file", paths["embeddings"],
                "--index_file",      paths["faiss_index"],
                "--meta_file",       paths["metadata"]
            ], "FAISS Indexer")
            st.sidebar.success("Vectors have been indexed to chunks")
            st.sidebar.success("✅ Full pipeline complete!")
            processed = True
        except Exception as e:
            st.sidebar.error(f"Invalid JSON or pipeline error: {e}")
else:
    st.sidebar.success("✅ Already processed")

# ─── Main: PDF Viewer & Query ──────────────────────────────────────────────
st.header(f"📖 {selected}")

# PDF Viewer → use expander instead of modal
with st.expander("👁 View PDF"):
    pdf_bytes = open(os.path.join(PDF_FOLDER, selected), "rb").read()
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    iframe = f'<iframe src="data:application/pdf;base64,{b64}#toolbar=0" width="100%" height="600px"></iframe>'
    st.components.v1.html(iframe, height=600)

# Query UI
if processed:
    query = st.text_input("Ask a question about this PDF")
    if st.button("Ask"):
        # embed
        q_emb = np.array(
            client.embeddings.create(input=[query], model=em_model)
                  .data[0]
                  .embedding,
            dtype="float32"
        )
        # retrieve
        idx  = faiss.read_index(paths["faiss_index"])
        meta = json.load(open(paths["metadata"], "r", encoding="utf-8"))
        D,I  = idx.search(q_emb.reshape(1,-1), top_k)
        hits = [meta[i] for i in I[0]]

        # context & cites
        context = "\n\n".join(h.get("text","") for h in hits)
        cites   = [
            f"§ {' > '.join(m.get('section_hierarchy',[]))} (p. {m['page_number']})"
            for m in hits
        ]

        # call LLM (no web-search)
        system = (
            "You are an AI assistant whose sole knowledge is the provided context. "
            "Only answer using information in that context—do not search the web. "
            "If you don't know, say so. List only provided citations."
        )
        user    = f"Context:\n{context}\n\nQuestion: {query}"
        resp    = client.chat.completions.create(
            model=llm_model,
            messages=[{"role":"system","content":system},
                    {"role":"user","content":user}]
        )
        answer  = resp.choices[0].message.content.strip()
        usage   = resp.usage
        st.session_state.prompt_tokens     += usage.prompt_tokens
        st.session_state.completion_tokens += usage.completion_tokens

        st.subheader("Answer")
        st.write(answer)
        st.subheader("Citations")
        for c in cites:
            st.write(c)
else:
    st.info("Run the pipeline to enable querying.")
