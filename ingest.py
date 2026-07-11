"""
ingest.py
---------
Loads data/gigacorp_faq.txt, splits it into Q&A chunks while preserving
the embedded doc-line numbers as metadata, embeds each chunk with
sentence-transformers/all-MiniLM-L6-v2 (via LangChain HuggingFaceEmbeddings),
and stores the resulting FAISS index under vectorstore/.

Usage:
    python ingest.py
"""

import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------
FAQ_PATH       = Path("data/gigacorp_faq.txt")
VECTORSTORE_DIR = Path("vectorstore")
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# 2. Parse the FAQ file into (doc_line_number, text) pairs
# ---------------------------------------------------------------------------

def parse_numbered_lines(filepath: Path) -> list[tuple[int, str]]:
    """
    Each physical line in the file starts with an embedded doc-line number
    (the number we want to cite), then one or more spaces, then the content.

    Example physical line:
        '11 A1.1: GigaCorp offers four shipping tiers ...'
         └─ doc line 11

    Returns a list of (doc_line_no, text_content) tuples, skipping
    blank-content lines (pure whitespace after stripping the number).
    """
    pattern = re.compile(r"^(\d+)\s+(.*)")
    results: list[tuple[int, str]] = []

    with open(filepath, encoding="utf-8") as fh:
        for physical_line in fh:
            m = pattern.match(physical_line)
            if m:
                doc_no  = int(m.group(1))
                content = m.group(2).rstrip()
                results.append((doc_no, content))

    return results


# ---------------------------------------------------------------------------
# 3. Group parsed lines into logical Q&A chunks
# ---------------------------------------------------------------------------

# Patterns that mark the start of a new chunk or a section header
_SECTION_HEADER = re.compile(r"^SECTION\s+\d+:", re.IGNORECASE)
_QA_QUESTION    = re.compile(r"^Q\d+\.\d+:")          # e.g. Q1.1:
_QA_ANSWER      = re.compile(r"^A\d+\.\d+:")          # e.g. A1.1:
_SEPARATOR      = re.compile(r"^-{10,}")               # dashed dividers
_DOC_META       = re.compile(                          # header / footer prose
    r"^(GIGACORP CUSTOMER FAQ|={5,}|Last Updated|Contact:|END OF DOCUMENT"
    r"|GigaCorp, Inc\.|gigacorp\.io)",
    re.IGNORECASE,
)

def build_chunks(parsed_lines: list[tuple[int, str]]) -> list[dict]:
    """
    Strategy
    --------
    Walk through the parsed lines and accumulate text into a chunk buffer.
    A new chunk is started whenever we encounter a Q-line (question).
    The current section name is tracked and injected into every chunk's
    metadata so the retriever knows which section the answer came from.

    Each returned dict has:
        {
            "text":       str,   # full Q+A text (possibly multi-line)
            "start_line": int,   # first doc-line number in this chunk
            "end_line":   int,   # last  doc-line number in this chunk
            "section":    str,   # e.g. "SECTION 1: SHIPPING POLICIES"
            "qa_id":      str,   # e.g. "Q1.2" (empty for non-Q&A chunks)
        }
    """
    chunks: list[dict] = []
    current_section = "PREAMBLE"
    current_chunk: list[tuple[int, str]] = []
    current_qa_id  = ""

    def flush(chunk_lines: list[tuple[int, str]], qa_id: str) -> None:
        if not chunk_lines:
            return
        text = "\n".join(c for _, c in chunk_lines).strip()
        if not text:
            return
        chunks.append({
            "text":       text,
            "start_line": chunk_lines[0][0],
            "end_line":   chunk_lines[-1][0],
            "section":    current_section,
            "qa_id":      qa_id,
        })

    for doc_no, content in parsed_lines:
        # Skip decorative separators and pure-whitespace lines
        if not content or _SEPARATOR.match(content):
            continue

        # Detect section headers — update running section, flush pending chunk
        if _SECTION_HEADER.match(content):
            flush(current_chunk, current_qa_id)
            current_chunk  = []
            current_qa_id  = ""
            current_section = content.strip()
            continue

        # Skip document metadata lines (title, footer, etc.)
        if _DOC_META.match(content):
            continue

        # Detect question start — flush previous chunk, begin new one
        if _QA_QUESTION.match(content):
            flush(current_chunk, current_qa_id)
            current_chunk = []
            m = re.match(r"^(Q\d+\.\d+):", content)
            current_qa_id = m.group(1) if m else ""

        # Accumulate lines into the current chunk
        current_chunk.append((doc_no, content))

    # Flush whatever remains
    flush(current_chunk, current_qa_id)
    return chunks


# ---------------------------------------------------------------------------
# 4. Convert chunks to LangChain Documents
# ---------------------------------------------------------------------------

def chunks_to_documents(chunks: list[dict]):
    """Wrap each chunk dict in a LangChain Document with rich metadata."""
    from langchain_core.documents import Document

    docs = []
    for chunk in chunks:
        metadata = {
            "source":     str(FAQ_PATH),
            "section":    chunk["section"],
            "qa_id":      chunk["qa_id"],
            "start_line": chunk["start_line"],
            "end_line":   chunk["end_line"],
            # Human-readable citation range for LLM responses
            "citation":   (
                f"gigacorp_faq.txt lines {chunk['start_line']}-{chunk['end_line']}"
            ),
        }
        docs.append(Document(page_content=chunk["text"], metadata=metadata))

    return docs


# ---------------------------------------------------------------------------
# 5. Embed + persist FAISS index
# ---------------------------------------------------------------------------

def build_and_save_index(docs) -> int:
    """Embed documents and save FAISS index. Returns number of docs stored."""
    from langchain_huggingface          import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    print(f"[ingest] Loading embedding model: {EMBED_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[ingest] Embedding {len(docs)} chunks ...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"[ingest] FAISS index saved to '{VECTORSTORE_DIR}/'")

    return len(docs)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not FAQ_PATH.exists():
        sys.exit(f"[ERROR] FAQ file not found: {FAQ_PATH}")

    print(f"[ingest] Parsing '{FAQ_PATH}' ...")
    parsed_lines = parse_numbered_lines(FAQ_PATH)
    print(f"[ingest] Read {len(parsed_lines)} numbered lines from the file.")

    print("[ingest] Building Q&A chunks ...")
    chunks = build_chunks(parsed_lines)

    # Debug preview — show every chunk's metadata
    for i, c in enumerate(chunks, 1):
        print(
            f"  Chunk {i:>2}: [{c['qa_id'] or 'narrative':>6}] "
            f"lines {c['start_line']:>3}-{c['end_line']:>3}  "
            f"| {c['section']}"
        )

    docs  = chunks_to_documents(chunks)
    count = build_and_save_index(docs)

    print(f"\nDone! {count} chunks indexed and saved to '{VECTORSTORE_DIR}/' successfully.")


if __name__ == "__main__":
    main()
