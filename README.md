# GigaCorp Support Assistant: Conversational RAG Engine

A production-grade, conversational Retrieval-Augmented Generation (RAG) assistant designed to answer customer queries with extreme precision, citing exact line numbers from GigaCorp's official FAQ documentation.

Built using **LangChain**, **FAISS**, **Hugging Face (`all-MiniLM-L6-v2`)**, **ChatGroq (`llama-3.3-70b-versatile`)**, and **Streamlit**.

---

## 🛠️ Architecture Overview

The system operates in two core pipelines: **Data Ingestion** and **Conversational Query Resolution**.

### 1. Ingestion Pipeline (`ingest.py`)
Parses, chunks, and vectorizes raw document data.

```
+---------------------------+
| data/gigacorp_faq.txt     | <--- Starts with physical doc-line numbers (1-182)
+---------------------------+
              |
              v (parse_numbered_lines)
+---------------------------+
| line number metadata maps |
+---------------------------+
              |
              v (build_chunks)
+---------------------------+
| Q&A Paragraph Blocks      | <--- Preserves (start_line, end_line) per chunk
+---------------------------+
              |
              v (HuggingFaceEmbeddings: all-MiniLM-L6-v2)
+---------------------------+
| Normalized Vector Spaces  |
+---------------------------+
              |
              v (FAISS)
+---------------------------+
| Vector Store (Local DB)   | <--- Saved under vectorstore/
+---------------------------+
```

### 2. Conversational RAG Pipeline (`rag_chain.py`)
Resolves multi-turn questions by contextualizing queries before retrieval.

```
       [ User Query ]
             |
             v
+----------------------------+
| Step 1: Condense Prompt    | <--- Rewrites "How much does it cost?" into
| (LLM + ChatMessageHistory) |      "How much does international shipping cost?"
+----------------------------+
             |
             v
+----------------------------+
| Step 2: Vector Search      | <--- Fetches top-3 nearest-neighbor Q&A chunks
| (FAISS Index)              |
+----------------------------+
             |
             v
+----------------------------+
| Step 3: Grounded Answer    | <--- LLM answers strictly from context and
| (LLM generation)           |      injects citations: e.g., [lines 10-14]
+----------------------------+
             |
             v
+----------------------------+
| Step 4: Metadata Parsing   | <--- Regex extracts cited ranges for UI badges
| (extract_cited_lines)      |      and appends to memory history
+----------------------------+
             |
             v
         [ Response ]
```

---

## 🔬 In-Depth Technical Details

### 1. Embedded Line-Level Citations
To guarantee verifiability, every physical line in [data/gigacorp_faq.txt](file:///c:/Users/acer/Desktop/Agent1/data/gigacorp_faq.txt) is prepended with its document line number.
During ingestion, `ingest.py` parses these prefixes out of the text, keeping track of the exact line ranges for each Q&A block. This metadata is saved inside the vector database as `start_line` and `end_line` keys:
```python
metadata = {
    "source": "data/gigacorp_faq.txt",
    "start_line": 50,
    "end_line": 56,
    "citation": "gigacorp_faq.txt lines 50-56"
}
```
When generating answers, the LLM is restricted by a system prompt instructing it to append the correct citation wrapper (e.g. `[lines 50-56]`) to any factual claim it presents. The chain then parses this string via regex to render interactive badge elements in the UI.

### 2. Context-Aware Query Condensing
In a standard RAG system, follow-up questions like *"How long does it take?"* yield poor vector retrieval because they contain pronouns and lack context. 
This engine resolves that by routing follow-up turns through a **Condenser LLM**. The condenser receives the running session transcript and outputs a self-contained query:
*   **User Turn 1:** *"Do you ship to India?"*
*   **User Turn 2:** *"How much does it cost?"* 
*   **Rewritten Standalone Query:** *"What is the cost of shipping to India from GigaCorp?"*

This standalone query is then used to query the FAISS index, ensuring the retriever fetches the correct international shipping policy block (lines 23-29) instead of general standard pricing.

### 3. Graceful Error Handling & Fallbacks
- **Auto-Ingestion:** The RAG chain checks for both `index.faiss` and `index.pkl`. If missing, it invokes the ingestion module dynamically to create them.
- **Secrets Resolution:** Checks system environment variables first, falling back to `st.secrets` on Streamlit Cloud.
- **Fail-Safe UI:** Initialization exceptions (e.g., missing API keys) are caught and displayed as a premium configuration error card instead of displaying raw redacted Streamlit Cloud tracebacks.

---

## 🚀 Setup & Execution Guide

### Local Installation
1. Clone this repository to your machine.
2. Install the pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Groq API Key:
   ```env
   GROQ_API_KEY=gsk_your_key_here
   ```

### Running the App
- **Compile the vector index manually (optional):**
  ```bash
  python ingest.py
  ```
- **Test semantic retrieval via CLI:**
  ```bash
  python verify_ingest.py
  ```
- **Launch the Streamlit App:**
  ```bash
  streamlit run app.py
  ```

### Streamlit Cloud Deployment
1. Push all code to GitHub. `.env` and `vectorstore/` will be ignored by Git to keep your API keys and build binaries private.
2. Deploy the repository on Streamlit Cloud.
3. Open **App Settings** -> **Secrets** in the lower-right console and paste your Groq API Key:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
4. Save the secret. Streamlit will automatically trigger a rebuild, auto-ingest the FAQ database, and start running the assistant.