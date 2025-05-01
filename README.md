# Grant-Compliance RAG System

End-to-end Retrieval-Augmented Generation pipeline for long policy PDFs.

## Features
- Universal font-inspector → robust header/body detection
- One-click pipeline: extract → split → embed → FAISS
- Streamlit UI with PDF viewer, model selector, token-cost meter
- Simple credential gate + secrets.toml support
- Completely offline RAG (no web search)

## Quick start

```bash
git clone <repo>
cd RAG_app
python -m venv .venv
.\.venv\Scripts\activate    # Win (use source .../activate on Mac/Linux)
pip install -r requirements.txt

# add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env          # or .streamlit/secrets.toml

streamlit run scripts/streamlit_app.py

---

## 🔧 Usage & Pipeline Tuning

### 1  PDF Management (sidebar ➊)

| Control | What it does |
|---------|--------------|
| **Upload PDF** | Adds a new file to `./pdfs/`. |
| **Select PDF** | Chooses an existing PDF to process / query. |
| **🗑 Delete** | Removes the selected PDF **and** its entire `./tmp/<name>/` folder (all embeddings, FAISS, etc.). |

---

### 2  Settings (sidebar ➋)

| Setting | Notes |
|---------|-------|
| **Embedding model** | Generally leave as default. Higher-end models slightly improve relevance but cost more. |
| **LLM model** | Choose the Chat model used to generate answers. Prices shown in the live cost meter come from `config.yaml ▶ cost:` |
| **Top-k retrievers** | How many chunks FAISS returns for your question.<br>• Lower ⇒ cheaper, faster, but may omit facts.<br>• Higher ⇒ more context, bigger prompt cost. |

---

### 3  Pipeline (sidebar ➌)

The pipeline runs in two phases so you can **approve the font mapping**:

1. **▶ Run Font Inspector**  
   *Scans every page, builds a font-size table + repeated-text counts, and writes*  
   `./tmp/<name>/font_inspector_output.json`.

2. **Review / Edit font mapping**  
   The JSON opens in a text area.  
   *You* decide for each `font_size` whether it is a **header**, **subheader**,  
   **body**, or should be **remove**d.  
   You can also add / delete lines in **`remove_texts`**.

   > **Tip – header rules**  
   > • Largest few sizes are usually *header / subheader*  
   > • Body sizes have the most `spans`  
   > • Repeated cover-page titles like “STATE OF MONTANA” are safe to mark **remove** if they show up on every page.

   Example edit for your snippet:

   ```jsonc
   {
     "font_size": 24.0,
     "classification": "remove"        // ok – appears only on cover
   },
   {
     "font_size": 27.96,
     "classification": "header"        // change from 'remove' → 'header'
   }

💰 Session usage — Prompt  823 tok, Completion  207 tok  •  Est. cost $0.031

### What you still might tweak
* Add more cost lines in `config.yaml` when you enable new models.  
* If the cover page has multiple *unique* large fonts you can leave their
  `classification` as **remove**—only mark **header** if that size is
  reused inside the document.

Commit this section to your README, push to GitHub, and your repository
now fully documents every control and how to fine-tune the
font-inspector output.
